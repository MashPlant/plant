use scoped_threadpool::Pool;
use xgboost::{parameters::{*, learning::*}, DMatrix, Booster};
use std::{mem, time::{Instant, Duration}, ops::{Deref, DerefMut, Index}, cmp::Ordering, fs::File, io::Write};
use crate::*;

#[derive(Debug)]
pub struct ConfigItem {
  pub name: R<str>,
  // values来源于Box<[u32]>，长度为len * each，逻辑上表示[[u32; each]; len]
  pub values: *mut u32,
  pub len: u32,
  pub each: u32,
}

impl ConfigItem {
  pub fn new(name: &str, values: Box<[u32]>, each: u32) -> Self {
    let len = values.len() as u32;
    debug_assert!(len != 0 && each != 0 && len % each == 0);
    ConfigItem { name: name.r(), values: Box::into_raw(values) as _, len: len / each, each }
  }

  pub fn get(&self, i: u32) -> &[u32] {
    debug_assert!(i < self.len);
    unsafe { std::slice::from_raw_parts(self.values.add(i as usize * self.each as usize), self.each as usize) }
  }

  pub fn values<'a>(&'a self) -> impl Display + 'a {
    comma_sep((0..self.len).map(move |i| fn2display(move |f| self.get(i).fmt(f))))
  }
}

impl Index<u32> for ConfigItem {
  type Output = [u32];
  fn index(&self, i: u32) -> &Self::Output { self.get(i) }
}

impl Drop for ConfigItem {
  fn drop(&mut self) {
    unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(self.values, self.len as usize * self.each as usize)); }
  }
}

pub struct ConfigSpace {
  // 搜索空间，每个元素表示选项的名字和选项的可能取值
  pub space: Vec<ConfigItem>,
  // 返回的Vec<P<Buf>>表示函数参数
  pub template: Box<dyn Fn(&ConfigEntity) -> (Vec<P<Buf>>, Box<Func>) + std::marker::Sync>,
}

impl ConfigSpace {
  // 返回Box<Self>是因为很多地方需要保存它的地址，ConfigSpace不能被移动
  pub fn new(template: impl Fn(&ConfigEntity) -> (Vec<P<Buf>>, Box<Func>) + std::marker::Sync + 'static) -> Box<Self> {
    box ConfigSpace { space: Vec::new(), template: box template }
  }

  // 搜索空间大小，即每个选项的可能取值数的积
  pub fn size(&self) -> u64 {
    self.iter().map(|i| i.len as u64).product()
  }

  // 返回本搜索空间上的一个随机的具体取值
  pub fn rand(&self, rng: &XorShiftRng) -> ConfigEntity {
    ConfigEntity { space: self.r(), choices: self.iter().map(|i| rng.gen() as u32 % i.len).collect() }
  }

  // 与rand类似，区别是往已经申请好的内存中填充随机值
  pub fn rand_fill(&self, rng: &XorShiftRng, choices: &mut Vec<u32>) {
    choices.clear();
    for i in self.iter() { choices.push(rng.gen() as u32 % i.len); }
  }
}

#[derive(Debug)]
pub struct SplitPolicy {
  pub n: u32,
  // pow2为true时考虑1, 2, ..., 2 ^ floor(log2(n))，否则考虑n的所有因子
  pub pow2: bool,
  pub allow_tail: bool,
  pub n_output: u32,
  pub max_factor: u32,
}

impl SplitPolicy {
  pub fn new(n: u32) -> Self {
    SplitPolicy { n, pow2: false, allow_tail: false, n_output: 2, max_factor: u32::MAX }
  }

  impl_setter!(set_pow2 pow2 bool);
  impl_setter!(set_allow_tail allow_tail bool);
  impl_setter!(set_n_output n_output u32);
  impl_setter!(set_max_factor max_factor u32);
}

macro_rules! impl_define {
  ($name: ident $ty: ty) => {
    pub fn $name(&self, name: &str, values: impl Into<Box<[$ty]>>) -> &Self {
      self.define_raw(name, unsafe { mem::transmute(values.into()) }, (mem::size_of::<$ty>() / mem::size_of::<u32>()) as _)
    }
  };
}

impl ConfigSpace {
  impl_define!(define u32);
  impl_define!(define2 (u32, u32));
  impl_define!(define3 (u32, u32, u32));
  impl_define!(define4 (u32, u32, u32, u32));
  impl_define!(define5 (u32, u32, u32, u32, u32));
  impl_define!(define6 (u32, u32, u32, u32, u32, u32));

  pub fn define_raw(&self, name: &str, values: Box<[u32]>, each: u32) -> &Self {
    let i = ConfigItem::new(name, values, each);
    info!("define_raw: name = {}, size = {}, values = [{}]", i.name, i.len, i.values());
    self.p().push(i);
    self
  }

  pub fn define_split(&self, name: &str, policy: P<SplitPolicy>) -> &Self {
    let n = policy.n;
    let factors = if policy.pow2 {
      (0..(31 - n.min(policy.max_factor).leading_zeros())).map(|x| 1 << x).collect::<Vec<_>>()
    } else {
      let mut factors = Vec::new();
      for i in (1..((n as f64).sqrt() as u32 + 1)).step_by((1 + n % 2) as usize) {
        if n % i == 0 {
          if i <= policy.max_factor { factors.push(i); }
          let j = n / i;
          if i != j && j <= policy.max_factor { factors.push(j); }
        }
      }
      factors.sort_unstable();
      factors
    };
    let (mut values, tmp) = (Vec::new(), vec![0; policy.n_output as usize - 1]);
    fn dfs(values: &mut Vec<u32>, tmp: &[u32], policy: P<SplitPolicy>, factors: &[u32], i: u32, prod: u32) {
      if let Some(x) = tmp.get(i as usize) {
        for &f in factors {
          *x.p() = f;
          dfs(values, tmp, policy, factors, i + 1, prod * f);
        }
      } else if prod <= policy.n && (policy.allow_tail || policy.n % prod == 0) {
        values.extend_from_slice(tmp);
      }
    }
    dfs(&mut values, &tmp, policy, &factors, 0, 1);
    self.define_raw(name, values.into(), policy.n_output - 1)
  }
}

impl Debug for ConfigSpace {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    f.debug_tuple("ConfigSpace").field(&self.space).finish()
  }
}

impl Deref for ConfigSpace {
  type Target = Vec<ConfigItem>;
  fn deref(&self) -> &Self::Target { &self.space }
}

impl DerefMut for ConfigSpace {
  fn deref_mut(&mut self) -> &mut Self::Target { &mut self.space }
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
  pub fn get(&self, name: &str) -> &[u32] {
    let (idx, i) = self.space.iter().enumerate().find(|(_, i)| &*i.name == name).unwrap();
    i.get(self.choices[idx])
  }
}

impl Index<&str> for ConfigEntity {
  type Output = u32;
  // 当取值唯一时，返回name选项的取值
  fn index(&self, name: &str) -> &Self::Output {
    let x = self.get(name);
    debug_assert_eq!(x.len(), 1);
    &x[0]
  }
}

impl Display for ConfigEntity {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    let mut m = f.debug_map();
    for (idx, i) in self.space.iter().enumerate() {
      m.entry(&i.name, &i.get(self.choices[idx]));
    }
    m.finish()
  }
}

pub struct TimeEvaluator {
  // 丢弃前n_discard次测试，至少会丢弃一次，即使n_discard是0
  pub n_discard: u32,
  // 计时重复n_repeat次
  pub n_repeat: u32,
  // 如果一次运行时长超过timeout，认为这个配置超时
  pub timeout: Duration,
  // 给运行函数提供输入，可以调用init设置为随机值，也可以手动设置为有意义的值，每个.1表示申请的字节数，drop时会释放这些内存
  // Func::codegen中生成的wrapper函数接受这样的指针p，从p[0], p[3], p[6], ...位置处读出实际函数的参数
  pub data: Option<Vec<Array<u8, usize>>>,
}

impl TimeEvaluator {
  pub fn new(n_discard: u32, n_repeat: u32, timeout: Duration) -> Self {
    TimeEvaluator { n_discard, n_repeat, timeout, data: None }
  }

  impl_setter!(set_n_discard n_discard u32);
  impl_setter!(set_n_repeat n_repeat u32);
  impl_setter!(set_timeout timeout Duration);
  impl_setter!(set_data data Option<Vec<Array<u8, usize>>>);

  // args和Func::codegen的args意义一样；init为args中每个Buf申请内存并用随机值初始化，保存在self.data中
  pub fn init(&self, args: &[P<Buf>]) {
    let rng = XorShiftRng(19260817);
    self.p().data = Some(args.iter().map(|&b| b.array(ArrayInit::Rand(&rng))).collect());
  }

  // 返回(耗时，!是否超时)，耗时单位为秒，如果超时了，耗时取一次运行的值
  // 如果发生运行错误，返回耗时inf
  pub fn eval(&self, f: WrapperFn) -> (f32, bool) {
    let data = self.data.as_ref()
      .expect("call TimeEvaluator::init or manually init TimeEvaluator::data first").as_ptr() as _;
    let t0 = Instant::now();
    // 预运行一次，用它判断是否超时
    f(data);
    let elapsed = Instant::now().duration_since(t0);
    if elapsed < self.timeout {
      // 预运行剩余次数
      for _ in 1..self.n_discard { f(data); }
      let t0 = Instant::now();
      for _ in 0..self.n_repeat { f(data); }
      (Instant::now().duration_since(t0).as_secs_f32() / self.n_repeat as f32, true)
    } else {
      (elapsed.as_secs_f32(), false)
    }
  }
}

pub struct Tuner {
  pub space: Box<ConfigSpace>,
  // 一次性编译运行batch_size个函数；编译是并行的(利用pool)，运行是串行的(因为需要计时)
  pub batch_size: u32,
  pub evaluator: TimeEvaluator,
  pub policy: TunerPolicy,
  // 记录当前最优的配置和这个配置下的耗时，单位为秒
  pub best: (ConfigEntity, f32),
  pub best_reporter: Option<Box<dyn FnMut(&(ConfigEntity, f32))>>,
  pub pool: Pool,
  // 理论上libs只是Tuner::eval中的局部变量，放在这里只是为了避免重复申请内存
  // 不能只保存WrapperFn而不保存Library，实际上存在生命周期的约束，后者析构后前者无法使用
  pub libs: Vec<Lib>,
}

pub fn file_best_reporter(beg: Instant, path: &str) -> impl FnMut(&(ConfigEntity, f32)) {
  let mut f = File::create(path).unwrap();
  move |(cfg, time)| {
    writeln!(f, "{:?}: best cfg {} time = {}s", Instant::now().duration_since(beg), cfg, time)
      .and_then(|_| f.flush()).unwrap();
  }
}

pub enum TunerPolicy {
  // 字典序搜索
  Search,
  Random(XorShiftRng),
  XGB(Box<XGBModel>),
}

impl Tuner {
  pub fn new(space: Box<ConfigSpace>, policy: TunerPolicy) -> Tuner {
    const DEFAULT_BATCH: u32 = 16;
    let best = (ConfigEntity { space: space.as_ref().r(), choices: <_>::default() }, f32::INFINITY);
    Tuner {
      space,
      batch_size: DEFAULT_BATCH,
      evaluator: TimeEvaluator::new(1, 3, Duration::from_secs(1)),
      policy,
      best,
      best_reporter: None,
      pool: Pool::new(DEFAULT_BATCH),
      libs: Vec::with_capacity(DEFAULT_BATCH as _),
    }
  }

  pub fn space(&self) -> R<ConfigSpace> { self.space.as_ref().r() }

  pub fn set_batch_size(&self, batch_size: u32) -> &Self {
    self.p().batch_size = batch_size;
    self.p().pool = Pool::new(batch_size);
    self.p().libs.reserve(batch_size as _);
    self
  }

  pub fn set_best_reporter(&self, f: impl FnMut(&(ConfigEntity, f32)) + 'static) {
    self.p().best_reporter = Some(box f);
  }

  // 尝试n_trial个配置取值
  pub fn tune(&self, n_trial: u32) {
    let mut batch = Vec::with_capacity(self.batch_size as usize);
    match &self.policy {
      Search => {
        struct Args<'a> { tuner: &'a Tuner, remain: u32, choices: Box<[u32]>, batch: Vec<ConfigEntity> }
        fn dfs(args: &mut Args, i: usize) {
          if args.remain == 0 { return; }
          if let Some(it) = args.tuner.space.get(i) {
            for x in 0..it.len {
              args.choices[i] = x;
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
          xgb.update(&batch, cost, &mut self.p().pool);
          batch.clear();
        }
      }
    }
  }

  // 若cost为Some，它应是和batch长度一样的slice，会把每个配置的耗时依次保存在其中
  pub fn eval(&self, batch: &[ConfigEntity], mut cost: Option<&mut [f32]>) {
    if let Some(cost) = &cost { debug_assert_eq!(cost.len(), batch.len()); }
    if self.evaluator.data.is_none() {
      // 默认在第一次运行前设置为随机值
      self.evaluator.init(&(self.space.template)(&batch[0]).0);
    }
    debug_assert_eq!(self.libs.len(), 0);
    self.p().libs.reserve(batch.len());
    unsafe { self.p().libs.set_len(batch.len()); }
    self.p().pool.scoped(|scope| {
      let (libs_ptr, template) = (P::new(self.libs.as_ptr()), &self.space.template);
      for (idx, cfg) in batch.iter().enumerate() {
        scope.execute(move || unsafe {
          let (bufs, f) = template(cfg);
          libs_ptr.0.as_ptr().add(idx).write(f.set_tmp(true).codegen(&bufs).unwrap());
        });
      }
    });
    for (idx, (lib, cfg)) in self.p().libs.drain(..).zip(batch.iter()).enumerate() {
      info!("before eval: {}", cfg); // 用于调试，最终还是没有找到捕获SIGSEGV的可靠方法，也许只能靠多进程，但我不愿意这样
      let (elapsed, ok) = self.evaluator.eval(lib.f);
      if ok {
        info!("eval: {}, {}s", cfg, elapsed);
      } else {
        warn!("eval: {} time out, {}s", cfg, elapsed);
      }
      if elapsed < self.best.1 {
        self.p().best.1 = elapsed;
        self.p().best.0 = cfg.clone();
      }
      if let Some(cost) = &mut cost { cost[idx] = elapsed; }
    }
    info!("eval: best cfg {} time = {}s", self.best.0, self.best.1);
    if let Some(x) = &mut self.p().best_reporter { x(&self.best); }
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

  // 用一组配置及其对应的耗时更新模型；cost表示耗时，单位无所谓(实际上Tuner传进来的单位是秒)
  pub fn update(&self, batch: &[ConfigEntity], cost: &[f32], pool: &mut Pool) {
    self.feature.get(batch, &mut self.p().xs, pool);
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
      info!("update: begin sa for, samples = {}, feature len = {}", self.xs_rows, self.xs.len());
      self.p().train_cnt += 1;
      self.sa(&self.model(), pool);
    }
  }

  pub fn model(&self) -> Booster {
    let mut dtrain = DMatrix::from_dense(&self.xs, self.xs_rows as _).unwrap();
    dtrain.set_labels(&self.ys).unwrap();
    self.p().params.set_dtrain(dtrain.p().get());
    Booster::train(&self.params).unwrap()
  }

  fn sa(&self, bst: &Booster, pool: &mut Pool) {
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
    self.feature.get(&points, &mut xs, pool);
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
        let val = self.rng.gen() as u32 % self.space[idx].len;
        p.choices[idx] = val;
        new_points.push(p);
      }
      xs.clear();
      self.feature.get(&new_points, &mut xs, pool);
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
