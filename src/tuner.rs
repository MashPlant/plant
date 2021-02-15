use libloading::{Library, Symbol};
use scoped_threadpool::Pool;
use xgboost::{parameters::{*, learning::*}, DMatrix, Booster};
use std::{mem, time::{Instant, Duration}, ops::{Deref, DerefMut}, cmp::Ordering};
use crate::*;

// 虽然有很多开源的随机数实现，但用自己的还是方便一点
#[derive(Debug, Clone, Copy)]
pub struct XorShiftRng(pub u64);

impl XorShiftRng {
  pub fn gen(&self) -> u64 {
    let mut x = self.p().get().0;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    self.p().get().0 = x;
    x
  }

  // 返回0.0~1.0间的浮点数
  pub fn gen_f32(&self) -> f32 {
    const INV_U32_MAX: f32 = 1.0 / u32::MAX as f32;
    self.gen() as u32 as f32 * INV_U32_MAX
  }
}

// 搜索空间，每个元素表示选项的名字和选项的可能取值
#[derive(Debug, Clone)]
pub struct ConfigSpace(pub Vec<(R<str>, Box<[u32]>)>);

impl ConfigSpace {
  // 返回Box<Self>是因为很多地方需要保存它的地址，ConfigSpace不能被移动
  pub fn new() -> Box<Self> { box ConfigSpace(Vec::new()) }

  // 搜索空间大小，即每个选项的可能取值数的积
  pub fn size(&self) -> u64 {
    self.iter().map(|(_, s)| s.len() as u64).product()
  }

  // 返回本搜索空间上的一个随机的具体取值
  pub fn rand(&self, rng: &XorShiftRng) -> ConfigEntity {
    ConfigEntity { space: self.r(), choices: self.iter().map(|(_, cs)| rng.gen() as u32 % cs.len() as u32).collect() }
  }

  // 与rand类似，区别是往已经申请好的内存中填充随机值
  pub fn rand_fill(&self, rng: &XorShiftRng, choices: &mut Vec<u32>) {
    choices.clear();
    for (_, cs) in self.iter() { choices.push(rng.gen() as u32 % cs.len() as u32); }
  }

  pub fn define(&self, name: &str, candidates: impl Into<Box<[u32]>>) -> &Self {
    let candidates = candidates.into();
    debug_assert!(!candidates.is_empty());
    self.p().push((name.into(), candidates));
    self
  }

  // 定义可能取值为1, 2, ..., 2 ^ floor(log2(n))的选项
  pub fn define_split_pow2(&self, name: &str, n: u32) -> &Self {
    let factors = (0..(31 - n.leading_zeros())).map(|x| 1 << x).collect::<Vec<_>>();
    info!("define_split_pow2: factors = {:?}", factors);
    self.define(name, factors)
  }

  // 定义可能取值n的所有因子的选项，因子不一定是按大小排序的
  pub fn define_split_factor(&self, name: &str, n: u32) -> &Self {
    let mut factors = Vec::new();
    for i in (1..((n as f64).sqrt() as u32 + 1)).step_by((1 + n % 2) as usize) {
      if n % i == 0 {
        factors.push(i);
        if i != n / i { factors.push(n / i); }
      }
    }
    info!("define_split_factor: factors = {:?}", factors);
    self.define(name, factors)
  }
}

impl Deref for ConfigSpace {
  type Target = Vec<(R<str>, Box<[u32]>)>;
  fn deref(&self) -> &Self::Target { &self.0 }
}

impl DerefMut for ConfigSpace {
  fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

// 搜索空间上一组具体的取值
#[derive(Clone, Debug)]
pub struct ConfigEntity {
  pub space: R<ConfigSpace>,
  // choices长度和space相同，每个元素表示space中对应元素的可能取值数组的下标
  pub choices: Box<[u32]>,
}

impl ConfigEntity {
  // 返回name选项的取值
  pub fn get(&self, name: &str) -> u32 {
    let (idx, (_, e)) = self.space.iter().enumerate().find(|(_, &(n, _))| &*n == name).unwrap();
    e[self.choices[idx] as usize]
  }
}

impl Display for ConfigEntity {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    let mut m = f.debug_map();
    for (idx, (name, candidates)) in self.space.iter().enumerate() {
      m.entry(name, &candidates[self.choices[idx] as usize]);
    }
    m.finish()
  }
}

pub struct Tuner {
  pub space: Box<ConfigSpace>,
  // 返回的Vec<P<Buf>>表示函数参数
  pub template: Box<dyn Fn(&ConfigEntity) -> (Vec<P<Buf>>, Box<Func>) + std::marker::Sync>,
  // 一次性编译运行batch_size个函数；编译是并行的(利用pool)，运行是串行的(因为需要计时)
  pub batch_size: u32,
  // 计时重复n_repeat次
  pub n_repeat: u32,
  // 如果一次运行时长超过timeout，认为这个配置超时
  pub timeout: Duration,
  pub policy: TunerPolicy,
  // 给运行函数提供输入，默认在第一次运行前设置为随机值，可以手动设置为有意义的值
  // Func::codegen中生成的wrapper函数接受这样的指针p，从p[0], p[2], p[4], ...位置处读出实际函数的参数
  pub data: Option<Vec<(*mut u8, usize)>>,
  // 记录当前最优的配置和这个配置下的耗时
  pub best: (ConfigEntity, Duration),
  pub pool: Pool,
  // 理论上libs只是Tuner::eval中的局部变量，放在这里只是为了避免重复申请内存
  pub libs: Vec<(Library, Symbol<'static, fn(*const *mut u8)>)>,
}

pub enum TunerPolicy {
  // 字典序搜索
  Search,
  Random(XorShiftRng),
  XGB(Box<XGBModel>),
}

impl Tuner {
  pub fn new(space: Box<ConfigSpace>, policy: TunerPolicy, template: impl Fn(&ConfigEntity) -> (Vec<P<Buf>>, Box<Func>) + std::marker::Sync + 'static) -> Tuner {
    const DEFAULT_BATCH: u32 = 16;
    // !0即无符号全1，相当于无穷大的耗时
    let best = (ConfigEntity { space: space.as_ref().r(), choices: <_>::default() }, Duration::from_secs(!0));
    Tuner {
      space,
      template: box template,
      batch_size: DEFAULT_BATCH,
      n_repeat: 3,
      timeout: Duration::from_secs(1),
      policy,
      data: None,
      best,
      pool: Pool::new(DEFAULT_BATCH),
      libs: Vec::with_capacity(DEFAULT_BATCH as _),
    }
  }

  pub fn space(&self) -> R<ConfigSpace> { self.space.as_ref().r() }

  impl_setter!(set_n_repeat n_repeat u32);
  impl_setter!(set_timeout timeout Duration);
  impl_setter!(set_policy policy TunerPolicy);
  impl_setter!(set_data data Option<Vec<(*mut u8, usize)>>);

  pub fn set_batch_size(&self, batch_size: u32) -> &Self {
    self.p().batch_size = batch_size;
    self.p().pool = Pool::new(batch_size);
    self.p().libs.reserve(batch_size as _);
    self
  }

  // 尝试n_trial个配置取值
  pub fn tune(&self, n_trial: u32) {
    let mut batch = Vec::with_capacity(self.batch_size as usize);
    match &self.policy {
      Search => {
        struct Args<'a> { tuner: &'a Tuner, remain: u32, choices: Box<[u32]>, batch: Vec<ConfigEntity> }
        fn dfs(args: &mut Args, i: usize) {
          if args.remain == 0 { return; }
          if let Some((_, candidates)) = args.tuner.space.get(i) {
            for x in 0..candidates.len() {
              args.choices[i] = x as _;
              dfs(args, i + 1);
            }
          } else {
            args.remain -= 1;
            args.batch.push(ConfigEntity { space: args.tuner.space(), choices: args.choices.clone() });
            if args.batch.len() as u32 == args.tuner.batch_size {
              args.tuner.eval(&args.batch, None);
              args.batch.clear();
            }
          }
        }
        let mut args = Args { tuner: self, remain: n_trial, choices: vec![0; self.space.len()].into(), batch };
        dfs(&mut args, 0);
        if !args.batch.is_empty() { self.eval(&args.batch, None); }
      }
      Random(rng) => {
        let mut i = 0;
        while i < n_trial {
          let n = self.batch_size.min(n_trial - i);
          i += n;
          for _ in 0..n { batch.push(self.space.rand(rng)); }
          self.eval(&batch, None);
          batch.clear();
        }
      }
      XGB(xgb) => {
        let mut i = 0;
        let mut cost = vec![0.0; self.batch_size as usize];
        while i < n_trial {
          let n = self.batch_size.min(n_trial - i);
          i += n;
          xgb.next_batch(&mut batch, n);
          let cost = &mut cost[..n as usize];
          self.eval(&batch, Some(cost));
          xgb.update(&batch, cost);
          batch.clear();
        }
      }
    }
  }

  // 若cost为Some，它应是和batch长度一样的slice，会把每个配置的耗时依次保存在其中
  pub fn eval(&self, batch: &[ConfigEntity], mut cost: Option<&mut [f32]>) {
    if let Some(cost) = cost.as_ref() { debug_assert_eq!(cost.len(), batch.len()); }
    let data = self.p().data.get_or_insert_with(|| {
      (self.template)(&batch[0]).0.iter().map(|b| {
        let mut size = b.ty.size();
        for s in &b.sizes {
          match s {
            &Val(ty, x) => size *= ty.val_i64(x) as usize,
            _ => debug_panic!("arg buf size must be Val"),
          }
        }
        info!("eval: buf {} alloc size = {}", b.name, size);
        (alloc::<u8>(size).as_ptr(), size)
      }).collect()
    }).as_ptr() as _;
    debug_assert_eq!(self.libs.len(), 0);
    self.p().libs.reserve(batch.len());
    unsafe { self.p().libs.set_len(batch.len()); }
    self.p().pool.scoped(|scope| {
      let (libs_ptr, template) = (P::new(self.libs.as_ptr()), &self.template);
      for (idx, cfg) in batch.iter().enumerate() {
        scope.execute(move || unsafe {
          let (bufs, f) = template(cfg);
          let lib = f.set_tmp(true).codegen(&bufs).unwrap();
          let f = (*(&lib as *const Library)).get(format!("{}_wrapper\0", f.name).as_bytes()).unwrap();
          libs_ptr.0.as_ptr().add(idx).write((lib, f));
        });
      }
    });
    for (idx, ((_, f), cfg)) in self.p().libs.drain(..).zip(batch.iter()).enumerate() {
      let t0 = Instant::now();
      // 预运行一次，不参与计时，只用它判断是否超时
      f(data);
      let t1 = Instant::now();
      let mut elapsed = t1.duration_since(t0);
      if elapsed < self.timeout {
        for _ in 0..self.n_repeat { f(data); }
        elapsed = Instant::now().duration_since(t1) / self.n_repeat;
        info!("eval: cfg {} time = {:?}", cfg, elapsed);
      } else {
        warn!("eval: cfg {} time out", cfg);
      }
      if elapsed < self.best.1 {
        self.p().best.1 = elapsed;
        self.p().best.0 = cfg.clone();
      }
      if let Some(cost) = cost.as_mut() { cost[idx] = elapsed.as_secs_f32(); }
    }
    info!("eval: best cfg {} time = {:?}", self.best.0, self.best.1);
  }
}

impl Drop for Tuner {
  fn drop(&mut self) {
    if let Some(data) = self.data.as_ref() {
      for &(p, size) in data { dealloc(p, size); }
    }
  }
}

pub struct XGBModel {
  pub space: R<ConfigSpace>,
  // 理论上libs只是XGBModel::update中的局部变量，放在这里只是为了避免重复申请内存
  // TrainingParameters的生命周期参数表示它借用的DMatrix，实际上它会引用XGBModel::update中的局部变量
  // 这里用'static是安全的，因为params只在Booster::train中使用，这个引用不会泄露到别的地方
  pub params: TrainingParameters<'static>,
  pub feature: Feature,
  // 已经访问过的点集，只有在XGBModel::next_batch中加入batch时才加入vis中
  pub vis: HashSet<Box<[u32]>>,
  // xs和ys保存所有的输入样本，每次都用所有样本训练一个新的模型(因此没有一个字段表示模型，模型只是局部变量)
  pub xs: Vec<f32>,
  pub ys: Vec<f32>,
  // 输入的样本数，xs长度为xs_rows * 每个样本的特征的长度
  pub xs_rows: u32,
  // 以下字段用于模拟退火，以下简称SA
  // 每新增plan_size个输入样本，进行一次SA，选择出plan_size个点(剩下的未访问的点较少时，有可能少于plan_size)，即plans
  pub plan_size: u32,
  // f32表示模型对这个配置的预测值，越大越好
  pub plans: Vec<(ConfigEntity, f32)>,
  // 当前访问到了plans的哪个下标
  pub plans_idx: u32,
  // 已经进行的模拟退火次数
  pub train_cnt: u32,
  // 与Tuner::batch_size，也即XGBModel::next_batch中传入的batch_size，意义不同
  // 这里表示SA过程中维护batch_size个点，一次性对它们进行修改和判定
  pub batch_size: u32,
  // SA进行的迭代轮数
  pub sa_iter: u32,
  // SA的温度设置，temp.0是初始温度，temp.1是最终温度，在sa_iter轮迭代内温度线性地降低
  pub temp: (f32, f32),
  pub rng: XorShiftRng,
  // 上次SA最后的点集(batch_size个)，作为下次SA的初始点集，一开始为空，用随机点集作为初始点集
  pub last_points: Vec<ConfigEntity>,
}

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum Loss { Reg, Rank }

// Feature表示从程序中提取特征的方法
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum Feature {
  // 直用每个选项的取值作为输入向量
  Knob,
  Iter,
  Curve,
}

impl Feature {
  // 把cfgs中配置的特征加入到xs中，不影响xs原来保存的内容
  pub fn get(self, cfgs: &[ConfigEntity], xs: &mut Vec<f32>) {
    match self {
      Feature::Knob => {
        xs.reserve(cfgs[0].space.len() * cfgs.len());
        for cfg in cfgs {
          for (idx, (_, candidates)) in cfg.space.iter().enumerate() {
            xs.push(candidates[cfg.choices[idx] as usize] as f32);
          }
        }
      }
      Feature::Iter => {}
      Feature::Curve => {}
    }
  }
}

impl XGBModel {
  pub fn new(space: &ConfigSpace, loss: Loss, feature: Feature) -> Box<Self> {
    let params = BoosterParameters::default();
    *params.learning_params().objective().p() = match loss { Loss::Reg => Objective::RegLinear, Loss::Rank => Objective::RankPairwise };
    box XGBModel {
      space: space.r(),
      params: TrainingParametersBuilder::default()
        // 构造TrainingParameters需要一个DMatrix的引用，但实际上不会访问它，这里没法提供这个引用，就用一个无效的地址
        // Booster::train中才会访问这个引用，在那之前已经设置成了有效的引用
        .dtrain(P::new(mem::align_of::<DMatrix>() as *const DMatrix).get())
        .boost_rounds(8000)
        .booster_params(params).build().unwrap(),
      feature,
      vis: <_>::default(),
      xs: Vec::new(),
      ys: Vec::new(),
      xs_rows: 0,
      plan_size: 64,
      plans: Vec::new(),
      plans_idx: 0,
      train_cnt: 0,
      batch_size: 128,
      sa_iter: 500,
      temp: (1.0, 0.0),
      rng: XorShiftRng(19260817),
      last_points: Vec::new(),
    }
  }

  impl_setter!(set_plan_size plan_size u32);
  impl_setter!(set_batch_size batch_size u32);
  impl_setter!(set_sa_iter sa_iter u32);
  impl_setter!(set_temp temp (f32, f32));

  // 通过self.plans和随机生成，往batch中填入batch_size个配置；如果剩下未访问的点太少，填入的配置可能少于batch_size
  pub fn next_batch(&self, batch: &mut Vec<ConfigEntity>, batch_size: u32) {
    let space_size = self.space.size();
    for _ in 0..batch_size {
      if self.vis.len() as u64 >= space_size { break; }
      let mut idx = self.plans_idx as usize;
      let mut choices = Vec::with_capacity(self.space.len());
      while let Some((cfg, _)) = self.plans.get(idx) {
        if !self.vis.contains(&cfg.choices) {
          choices.extend_from_slice(&cfg.choices);
          break;
        }
        idx += 1;
      }
      self.p().plans_idx = idx as _;
      // eps-greedy策略，舍弃最后5%(排在后面的预测值低)的点，随机选择
      // 第一次调用next_batch时self.plans为空，也会进入这个分支
      if idx + (self.plan_size / 20) as usize >= self.plans.len() {
        loop {
          self.space.rand_fill(&self.rng, &mut choices);
          if !self.vis.contains(choices.as_slice()) { break; }
        }
      }
      debug_assert!(!choices.is_empty());
      let choices = choices.into_boxed_slice();
      self.p().vis.insert(choices.clone());
      batch.push(ConfigEntity { space: self.space, choices });
    }
  }

  // 用一组配置及其对应的耗时更新模型；cost表示耗时，单位无所谓
  pub fn update(&self, batch: &[ConfigEntity], cost: &[f32]) {
    self.feature.get(batch, &mut self.p().xs);
    self.p().xs_rows += batch.len() as u32;
    self.p().ys.reserve(cost.len());
    let mut cost_min = 1e9;
    // 不能用Iterator::min，因为f32没有实现Ord
    for &cost in cost { cost_min = cost.min(cost_min); }
    // 归一化，cost最少的y为1，其余都小于1，所以模型预测的值越大越好
    for &cost in cost { self.p().ys.push(cost_min / cost); }
    debug_assert_eq!(self.xs_rows as usize, self.ys.len());
    // 新增样本数达到了plan_size，进行一次SA，选择之后要给出的plans
    if self.xs_rows >= (self.train_cnt + 1) * self.plan_size {
      info!("update: begin sa for {} samples", self.xs_rows);
      self.p().train_cnt += 1;
      self.sa(&self.model());
    }
  }

  pub fn model(&self) -> Booster {
    let mut dtrain = DMatrix::from_dense(&self.xs, self.xs_rows as _).unwrap();
    dtrain.set_labels(&self.ys).unwrap();
    self.p().params.set_dtrain(dtrain.p().get());
    Booster::train(&self.params).unwrap()
  }

  fn sa(&self, bst: &Booster) {
    let plans = &mut self.p().plans;
    plans.clear();
    // 初始时往plans中填入plan_size个无效的值，choices为空，预测值为-inf
    for _ in 0..self.plan_size {
      plans.push((ConfigEntity { space: self.space.r(), choices: <_>::default() }, f32::NEG_INFINITY));
    }
    // 第一次last_points为空，用随机点集作为初始点集，之后使用上次留下的last_points
    let mut points = if self.last_points.is_empty() {
      (0..self.batch_size).map(|_| self.space.rand(&self.rng)).collect()
    } else {
      mem::replace(&mut self.p().last_points, Vec::new())
    };
    let mut xs = Vec::new();
    self.feature.get(&points, &mut xs);
    let mut ys = bst.predict(&DMatrix::from_dense(&xs, points.len()).unwrap()).unwrap();
    fn update_plans(points: &[ConfigEntity], ys: &[f32], plans: &mut Vec<(ConfigEntity, f32)>, vis: &HashSet<Box<[u32]>>) {
      debug_assert_eq!(points.len(), ys.len());
      for (p, &y) in points.iter().zip(ys) {
        // 如果p不在vis中，且p不在plans的点中，且plans中最小的预测值比p的预测值小，则用p替换最小的预测值对应的点
        if !vis.contains(&p.choices) {
          let (mut min_idx, mut min) = (!0, y); // !0表示无效
          for (idx, (p1, y1)) in plans.iter().enumerate() {
            if p1.choices == p.choices {
              min_idx = !0; // p在plans的点中，立即失败
              break;
            }
            if *y1 < min {
              min = *y1;
              min_idx = idx;
            }
          }
          if let Some(x) = plans.get_mut(min_idx) { *x = (p.clone(), y); }
        }
      }
    }
    update_plans(&points, &ys, plans, &self.vis);
    let mut temp = self.temp.0;
    let cool = 1.0 * (self.temp.0 - self.temp.1) / (self.sa_iter + 1) as f32;
    let mut new_points = Vec::with_capacity(points.len());
    for _ in 0..self.sa_iter {
      for p in &points {
        let mut p = p.clone();
        // 随机选择一个选项，修改成随机的一个可能值
        let idx = self.rng.gen() as usize % self.space.len();
        let val = self.rng.gen() as u32 % self.space[idx].1.len() as u32;
        p.choices[idx] = val;
        new_points.push(p);
      }
      xs.clear();
      self.feature.get(&new_points, &mut xs);
      let new_ys = bst.predict(&DMatrix::from_dense(&xs, new_points.len()).unwrap()).unwrap();
      update_plans(&new_points, &new_ys, plans, &self.vis);
      debug_assert_eq!(points.len(), new_points.len());
      for (((p, y), new_p), &new_y) in points.iter_mut().zip(ys.iter_mut())
        .zip(new_points.drain(..)).zip(new_ys.iter()) {
        let prob = ((new_y - *y) / (temp + 1e-5)).exp();
        if self.rng.gen_f32() < prob {
          *p = new_p;
          *y = new_y;
        }
      }
      temp -= cool;
    }
    // 按预测值从大到小排序。同样因为f32没有实现Ord，必须用这种间接的写法
    plans.sort_unstable_by(|&(_, y1), &(_, y2)|
      if y1 > y2 { Ordering::Less } else if y1 < y2 { Ordering::Greater } else { Ordering::Equal });
    // 负值可能是最初填入的-inf或者预测为负的有效值(尽管训练模型时输入都在0~1间，模型还是会得到超出这个范围的结果)
    // 现在它们(如果存在)排在后面，-inf是无效的值，负的有效值的期望效果也很差，把它们删掉
    if let Some(last_valid) = plans.iter().position(|&(_, y)| y < 0.0) {
      plans.drain(last_valid..);
    }
    self.p().plans_idx = 0;
    self.p().last_points = points;
  }
}
