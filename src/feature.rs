use scoped_threadpool::Pool;
use std::cmp::Ordering;
use crate::*;

// 从程序中提取特征的方法
#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub enum Feature {
  // 直用每个选项的取值作为输入向量
  Knob,
  // 从循环变量中提取特征，bool表示是否对特征数值取log
  Iter(bool),
  // TVM中的解释是sampled curve feature (relation feature)，我只是照着实现了，并没有看懂原理
  // u32表示采样点数目
  Curve(u32),
}

pub fn default_iter() -> Feature { Feature::Iter(true) }

pub fn default_curve() -> Feature { Curve(20) }

impl Feature {
  // 把cfgs中配置的特征加入到xs中，不影响xs原来保存的内容
  pub fn get(self, cfgs: &[ConfigEntity], xs: &mut Vec<f32>, pool: &mut Pool) -> Unit {
    // Iter和Curve要求程序结构不变，只是数值变化，才能保证每个特征向量的长度一样
    // _check_len和下面的_old_len用来检查特征向量的长度确实是一样的
    let mut _check_len = None;
    macro_rules! check_len {
      ($old_len: ident) => {
        if cfg!(debug_assertions) {
          assert!(_check_len.is_none() || _check_len == Some(xs.len() - $old_len));
          _check_len = Some(xs.len() - $old_len);
        }
      };
    }
    let mut gen_iters = move || {
      let th = pool.thread_count() as usize;
      let mut iters = vec![(vec![], 1); th];
      let (div, rem) = (cfgs.len() / th, cfgs.len() % th);
      pool.scoped(|scope| {
        let mut beg = 0;
        for (idx, result) in iters.iter_mut().enumerate() {
          let end = beg + div + (idx < rem) as usize;
          debug_assert!(end <= cfgs.len());
          let cfgs = &cfgs[beg..end];
          beg = end;
          if !cfgs.is_empty() {
            scope.execute(move || {
              let mut ext = TouchExtractor::default();
              for cfg in cfgs { ext.extract(cfg); }
              let size = ext.iter_result.len() / cfgs.len();
              *result = (ext.iter_result, size);
            });
          }
        }
        debug_assert_eq!(beg, cfgs.len());
      });
      iters
    };
    match self {
      Feature::Knob => {
        xs.reserve(cfgs[0].space.len() * cfgs.len());
        for cfg in cfgs {
          for (idx, (_, candidates)) in cfg.space.iter().enumerate() {
            xs.push(candidates[cfg.choices[idx] as usize] as f32);
          }
        }
      }
      Feature::Iter(log) => {
        let trans = move |x: i64| if log { ((x as f64).abs() + 1.0).log2() as f32 } else { x as f32 };
        for iter in gen_iters().iter().flat_map(|(iters, size)| iters.chunks(*size)) {
          let _old_len = xs.len();
          for fea in iter {
            xs.push(trans(fea.extent));
            xs.push(fea.nest_level as _);
            // one-hot编码tag，tag为None时编码为[1, 0, 0, ...]，tag为Some(Parallel)时编码为[0, 1, 0, ...]，以此类推
            for i in 0..=GPUThreadZ as u32 + 1 {
              xs.push(if let Some(tag) = fea.tag { i == tag as u32 + 1 } else { i == 0 } as u32 as _);
            }
            xs.push(trans(fea.topdown_product));
            xs.push(trans(fea.bottomup_product));
            xs.push(trans(fea.arith_cnts[0] as _));
            xs.push(trans(fea.arith_cnts[1] as _));
            xs.push(trans(fea.arith_cnts[2] as _));
            for pat in fea.touch_pattern.values() {
              xs.push(trans(pat.stride));
              xs.push(trans(pat.count));
              xs.push(trans(pat.reuse));
              xs.push(trans(pat.thread_count));
              xs.push(trans(pat.thread_reuse));
            }
          }
          check_len!(_old_len);
        }
      }
      Feature::Curve(sample) => {
        // 定义放在循环外减少内存申请
        let (mut count_curve, mut reuse_curve, mut topdown_curve) = (HashMap::default(), HashMap::default(), HashMap::default());
        let mut inner_bufs = Vec::new();
        let mut added = HashSet::default();
        for iter in gen_iters().iter().flat_map(|(iters, size)| iters.chunks(*size)) {
          let _old_len = xs.len();
          (count_curve.clear(), reuse_curve.clear(), topdown_curve.clear(), inner_bufs.clear(), added.clear());
          // 优先找最深的循环，再找extent尽量大的
          let (max_depth, max_extent) = iter.iter().map(|fea| (fea.nest_level, fea.extent)).max().unwrap_or((0, 0));
          for fea in iter {
            if fea.nest_level == max_depth && fea.extent == max_extent {
              for buf in fea.touch_pattern.keys() {
                // 对一个Buf只记录一次访问
                if added.insert((buf.0).0) { inner_bufs.push(((buf.0).0, buf.1)); }
              }
            }
          }
          // 放个0占位，这样sample_curve总会进入if分支一次
          // 这些数据结构不需要稳定的顺序，所以把NameHashBuf中的P<Buf>取出来
          for &buf in &inner_bufs {
            count_curve.insert(buf, vec![0.0]);
            reuse_curve.insert(buf, vec![0.0]);
            topdown_curve.insert(buf, vec![0.0]);
          }
          for fea in iter {
            for (buf, pat) in &fea.touch_pattern {
              let ref buf = ((buf.0).0, buf.1);
              if inner_bufs.contains(buf) {
                count_curve.get_mut(buf)?.push((pat.count as f64).log2());
                reuse_curve.get_mut(buf)?.push((pat.reuse as f64).log2());
                topdown_curve.get_mut(buf)?.push((fea.topdown_product as f64).log2());
              }
            }
          }
          let sample_curve = move |xs: &mut Vec<f32>, x: &[f64], y: &[f64]| {
            let _old_len = xs.len();
            for i in 0..sample {
              let i = i as f64;
              for (&x, &y) in x.iter().zip(y.iter()).rev() {
                if i - x > -1e-6 {
                  xs.push(y as _);
                  xs.push((i - x) as _);
                  break;
                }
              }
            }
            // 每轮for都一定会进入if分支一次，push两个值
            debug_assert_eq!(_old_len + sample as usize * 2, xs.len());
          };
          for buf in &inner_bufs {
            let (count, reuse, topdown) = (count_curve.get_mut(buf)?, reuse_curve.get_mut(buf)?, topdown_curve.get_mut(buf)?);
            // 只是正常的比较逻辑，f64没有实现Ord，必须自己提供比较函数
            let cmp = |l: &f64, r: &f64| if l < r { Ordering::Less } else if l > r { Ordering::Greater } else { Ordering::Equal };
            count.sort_unstable_by(cmp);
            reuse.sort_unstable_by(cmp);
            topdown.sort_unstable_by(cmp);
            sample_curve(xs, count, reuse);
            sample_curve(xs, reuse, count);
            sample_curve(xs, count, topdown);
            sample_curve(xs, topdown, count);
          }
          check_len!(_old_len);
        }
      }
    }
    Unit
  }
}

#[derive(Clone, Debug)]
pub struct TouchPattern {
  pub stride: i64,
  pub count: i64,
  pub reuse: i64,
  pub thread_count: i64,
  pub thread_reuse: i64,
}

#[derive(Clone, Debug)]
pub struct IterFeature {
  // extent即max - min + 1
  pub extent: i64,
  pub nest_level: u32,
  pub tag: Option<DimTag>,
  // 从上到下每个循环的extent的乘积
  pub topdown_product: i64,
  // bottomup_product = max(pat.reuse * pat.count) for pat in touch_pattern.values()
  pub bottomup_product: i64,
  // 统计这个循环层次中的[加法，乘法，除法]次数
  pub arith_cnts: [u32; 3],
  // 表示对Buf的一次访问，u32来自TouchExtractor::buf_cnt
  pub touch_pattern: HashMap<(NameHashBuf, u32), TouchPattern>,
}

#[derive(Default)]
pub struct TouchExtractor {
  // 来自CodegenState::info
  pub info: HashMap<AstNodeRef, ForInfo>,
  // 处理一个循环时，只需要考虑包裹它的循环，用栈结构保存这些循环
  // 循环不一定是完美嵌套的，所有循环的信息都需要保存下来，在循环弹出栈后将它push到iter_result中
  pub iter_stack: Vec<IterFeature>,
  // 按后序遍历顺序保存循环
  pub iter_result: Vec<IterFeature>,
  // 从上到下累乘每个循环的extent
  pub topdown_product: i64,
  // 为每个Buf维护一个计数器，以区分对一个Buf的多次访问
  pub buf_cnt: HashMap<P<Buf>, u32>,
}

pub fn parallel_level(tag: Option<DimTag>) -> u32 {
  if let Some(tag) = tag {
    match tag {
      GPUBlockX | GPUBlockY | GPUBlockZ => 2,
      GPUThreadX | GPUThreadY | GPUThreadZ | Parallel => 1,
      Vectorize | Unroll => 0,
    }
  } else { 0 }
}

impl TouchExtractor {
  // 不改变iter_result中原有的内容，如果有需要，用户自己负责清空
  pub fn extract(&mut self, cfg: &ConfigEntity) {
    let (_, f) = (cfg.space.template)(&cfg);
    let AstInfo(n, s, _i) = f.build_ast();
    self.info = s.info;
    debug_assert!(self.iter_stack.is_empty());
    self.topdown_product = 1;
    self.buf_cnt.clear();
    self.node(&f, *n);
  }

  pub fn expr(&mut self, e: &Expr) {
    e.visit(&mut move |e| match e {
      Binary(op, _) => {
        use BinOp::*;
        let idx = match op { Add | Sub => 0, Mul => 1, Div | Rem => 2, _ => return };
        self.iter_stack.last_mut().unwrap().arith_cnts[idx] += 1;
      }
      Load(buf, idx) => self.mem(*buf, idx),
      _ => {}
    })
  }

  pub fn mem(&mut self, buf: P<Buf>, idx: &Expr) {
    let buf_id = *self.buf_cnt.entry(buf).and_modify(|x| *x += 1).or_insert(0);
    let mut stride_map = vec![0; self.iter_stack.len()];
    // 这里不用e.visit，它不方便维护stride这样的参数
    fn vis_idx(stride_map: &mut [i64], e: &Expr, stride: i64) {
      match e {
        // 如果下标中一个循环变量出现多次，取stride绝对值最大的一次
        &Iter(_, it) => if stride_map[it as usize].abs() < stride.abs() { stride_map[it as usize] = stride; }
        Binary(BinOp::Mul, box [Val(ty, x), y]) | Binary(BinOp::Mul, box [y, Val(ty, x)]) =>
          vis_idx(stride_map, y, stride * ty.val_i64(*x)),
        _ => for x in e.args() { vis_idx(stride_map, x, stride); }
      }
    }
    vis_idx(&mut stride_map, idx, 1);
    // 包裹这次内存访问的所有循环变量都与这次访问有关，如果idx中没有这个循环变量，stride就是0
    for (it, &stride) in self.iter_stack.iter_mut().zip(stride_map.iter()) {
      it.touch_pattern.insert((NameHashBuf(buf.p()), buf_id),
        TouchPattern { stride, count: 1, reuse: 1, thread_count: 0, thread_reuse: 0 });
    }
  }

  pub fn node(&mut self, f: &Func, n: AstNodeRef) -> Unit {
    match n.get_type() {
      AstNodeType::For => {
        let info = self.info.get(&n)?;
        let extent = info.extent().2.get_num_si();
        let old_product = self.topdown_product;
        self.topdown_product *= extent;
        self.iter_stack.push(IterFeature {
          extent,
          nest_level: self.iter_stack.len() as _,
          tag: info.tag,
          topdown_product: self.topdown_product,
          bottomup_product: 0, // 之后再设置
          arith_cnts: [0; 3],
          touch_pattern: <_>::default(),
        });
        self.expr(&Expr::from_isl(f, n.for_get_init()?));
        self.expr(&Expr::from_isl(f, n.for_get_cond()?));
        self.expr(&Expr::from_isl(f, n.for_get_inc()?));
        self.node(f, *n.for_get_body()?);
        self.topdown_product = old_product;
        let cur = self.iter_stack.last()?.p();
        // 依据本层循环访问的内存和循环extent更新所有包裹它的循环和它自身的内存访问pattern
        for (buf, pat) in &cur.touch_pattern {
          // 这个循环包括cur在内(这违反借用规则，所以上面用了.p())
          for fea in &mut self.iter_stack {
            let pat1 = fea.touch_pattern.get_mut(buf)?;
            // 某个循环变量在下标中的stride为0，意味着内存访问与它的值无关，所以是reuse
            *if pat.stride == 0 { &mut pat1.reuse } else { &mut pat1.count } *= cur.extent;
          }
        }
        let mut cur = self.iter_stack.pop()?;
        cur.bottomup_product = cur.touch_pattern.values().map(|pat| pat.count * pat.reuse).max().unwrap_or(-1);
        let level = parallel_level(cur.tag);
        // 在并行等级的分界线处，用本层的count/reuse更新上层的thread_count/thread_reuse
        if self.iter_stack.last().filter(|fea| parallel_level(fea.tag) == level + 1).is_some() {
          for (buf, pat) in &cur.touch_pattern {
            for fea in &mut self.iter_stack {
              if parallel_level(fea.tag) == level + 1 {
                let pat1 = fea.touch_pattern.get_mut(buf)?;
                // 负号表示并非最终结果，处理到这一层时的thread_count/thread_reuse才是最终有效的，在此之前可能赋值多次
                pat1.thread_count = -pat.count;
                pat1.thread_reuse = -pat.reuse;
              }
            }
          }
        }
        for pat in cur.touch_pattern.values_mut() {
          if pat.thread_count < 0 {
            pat.thread_count = pat.count / (-pat.thread_count);
            pat.thread_reuse = pat.reuse / (-pat.thread_reuse);
          }
        }
        self.iter_result.push(cur);
      }
      AstNodeType::If => {
        self.expr(&Expr::from_isl(f, n.if_get_cond()?));
        self.node(f, *n.if_get_then()?);
        if let Some(e) = n.if_get_else() { self.node(f, *e); }
      }
      AstNodeType::Block => {
        let ch = n.block_get_children()?;
        for i in 0..ch.n_ast_node() { self.node(f, *ch.get_ast_node(i)?)?; }
      }
      AstNodeType::User => {
        let comp = P::<CompInfo>::new(n.get_annotation()?.get_user() as _);
        self.expr(&comp.expr);
        // store是用Load表示的Expr，最终会调用self.mem
        if let Some(s) = &comp.store { self.expr(s); }
      }
      ty => debug_panic!("invalid ast node type: {:?}", ty),
    }
    Unit
  }
}
