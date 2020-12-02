extern crate simdeez;

#[cfg(test)]
mod tests {    
    use simdeez::avx2::*;
    use simdeez::scalar::*;
    use simdeez::sse2::*;
    use simdeez::sse41::*;
    use simdeez::*;
    use simdeez::special::SimdSpecial;
    use std::f32::*;
    use std::f64::*;
    use std::*;
    use float_cmp::Ulps;

    // Macro for checking if f32/f64 are equal to within a delta
    macro_rules! assert_delta {
        ($x:expr, $y:expr, $d:expr) => {
            if $x.is_nan() && $y.is_nan() {
            } else if $x.is_nan() {
                assert!(false, "{} != NaN", $y);
            } else if $y.is_nan() {
                assert!(false, "{} != NaN", $x);
            } else if $x.is_infinite() && $y.is_infinite() {
            } else if $x.is_infinite() {
                assert!(false, "{} != infinity", $y);
            } else if $y.is_infinite() {
                assert!(false, "{} != infinity", $x);
            } else {
                assert!(($x - $y).abs() < $d, "{} != {}", $x, $y);
            }
        };
    }

    simd_runtime_generate!(
        fn set1(floats: &Vec<f32>, ints: &Vec<i32>) {
            for i in 0..ints.len() {
                let a = S::set1_epi32(ints[i]);
                for j in 0..S::VI32_WIDTH {
                    assert_eq!(a[j], ints[i]);
                }
            }
            for i in 0..floats.len() {
                let b = S::set1_ps(floats[i]);
                for j in 0..S::VF32_WIDTH {
                    assert_delta!(b[j], floats[i], 0.001);
                }
            }
        }
    );
    #[test]
    fn set1_test() {
        let ints = &vec![-1, 1, 0, 10, -10, i32::max_value(), i32::min_value()];
        let floats = &vec![
            -1.0,
            1.0,
            0.0,
            -0.0,
            10.0,
            -10.0,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ];
        unsafe {
            set1_sse2(floats, ints);
            set1_sse41(floats, ints);
            set1_avx2(floats, ints);            
            set1_scalar(floats, ints);
        }
    }
    simd_runtime_generate!(
        fn sub(floats: &Vec<f32>, ints: &Vec<i32>) {
            for &int1 in ints {
                for &int2 in ints {
                    let a = S::sub_epi32(S::set1_epi32(int1), S::set1_epi32(int2));
                    let expected = int1.wrapping_sub(int2);
                    for j in 0..S::VI32_WIDTH {
                        assert_eq!(a[j], expected, "{} - {} != {}", int1, int2, a[j]);
                    }
                }
            }
            for &float1 in floats {
                for &float2 in floats {
                    let b = S::sub_ps(S::set1_ps(float1), S::set1_ps(float2));
                    for j in 0..S::VF32_WIDTH {
                        assert_delta!(b[j], float1 - float2, 0.001);
                    }
                }
            }
        }
    );
    #[test]
    fn sub_test() {
        let ints = &vec![-1, 1, 0, 10, -10, i32::max_value(), i32::min_value()];
        let floats = &vec![
            -1.0,
            1.0,
            0.0,
            -0.0,
            10.0,
            -10.0,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ];
        unsafe {
            sub_sse2(floats, ints);
            sub_sse41(floats, ints);            
            sub_avx2(floats, ints);
            sub_scalar(floats, ints);
        }
    }
    simd_compiletime_generate!(
        fn cvt(floats: &Vec<f32>, ints: &Vec<i32>) {
            for i in 0..ints.len() {
                let a = S::cvtepi32_ps(S::set1_epi32(ints[i]));
                for j in 0..S::VI32_WIDTH {
                    assert_delta!(a[j], ints[i] as f32, 0.001);
                }
            }
            for i in 0..floats.len() {
                let b = S::cvtps_epi32(S::set1_ps(floats[i]));
                for j in 0..S::VF32_WIDTH {
                    println!("i:{}", i);
                    let x = floats[i];
                    let rounded = (x + 0.5).floor();
                    assert_eq!(b[j], rounded as i32);
                }
            }
        }
    );
    #[test]
    fn cvt_test() {
        let ints = &vec![-1, 1, 0, 10, -10, i32::max_value(), i32::min_value()];
        let floats = &vec![
            1.5,
            -1.5,
            -0.5,
            0.5,
            -0.999,
            0.999,
            0.0001,
            -0.0001,
            -1.0,
            1.0,
            0.0,
            -0.0,
            10.0,
            -10.0,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ];
        unsafe {
            //cvt_compiletime(floats, ints);
            //        cvt_sse2(floats,ints);
            //        cvt_sse41(floats,ints);
            //        cvt_avx2(floats,ints);
            //        cvt_scalar(floats,ints);
        }
    }
    simd_runtime_generate!(
        fn blendv() {
            //i32
            let a = S::set1_epi32(1);
            let b = S::set1_epi32(2);
            let cmp = S::cmplt_epi32(a, b);
            let r = S::blendv_epi32(a, b, cmp);
            //r should be all 2
            for i in 0..S::VI32_WIDTH {
                assert_eq!(r[i], 2);
            }

            let a = S::set1_epi32(2);
            let b = S::set1_epi32(1);
            let cmp = S::cmplt_epi32(a, b);
            let r = S::blendv_epi32(a, b, cmp);
            //r should be all 2
            for i in 0..S::VI32_WIDTH {
                assert_eq!(r[i], 2);
            }
            
            let a = S::set1_ps(2.0);
            let b = S::set1_ps(1.0);
            let cmp = S::cmplt_ps(a, b);
            let r = S::blendv_ps(a, b, cmp);
            //r should be all 2
            for i in 0..S::VI32_WIDTH {
                assert_eq!(r[i], 2.0);
            }
            let a = S::set1_ps(1.0);
            let b = S::set1_ps(9.0);
            let cmp = S::set1_epi32(-1);
            let r = S::blendv_ps(a, b, S::castepi32_ps(cmp));
            //r should be all 9

            for i in 0..S::VI32_WIDTH {
                assert_eq!(r[i], 9.0);
            }

            //f64
            let a = S::set1_pd(1.0);
            let b = S::set1_pd(2.0);
            let cmp = S::cmplt_pd(a, b);
            let r = S::blendv_pd(a, b, cmp);
            //r should be all 2
            for i in 0..S::VI64_WIDTH {
                assert_eq!(r[i], 2.0);
            }

            let a = S::set1_pd(2.0);
            let b = S::set1_pd(1.0);
            let cmp = S::cmplt_pd(a, b);
            let r = S::blendv_pd(a, b, cmp);
            //r should be all 2
            for i in 0..S::VI64_WIDTH {
                assert_eq!(r[i], 2.0);
            }
        }
    );

    #[test]
    fn blendv_test() {
        unsafe {
            blendv_sse41();
            blendv_avx2();            
            blendv_sse2();
            blendv_scalar();
        }
    }
    
    simd_runtime_generate!(
        fn sum_simdeez_horizontal(x: &[f32]) -> f32 {
            assert!(x.len() % S::VF32_WIDTH == 0);

            S::horizontal_add_ps(
                x.chunks_exact(S::VF32_WIDTH)
                    .map(|y| S::loadu_ps(&y[0]))
                    .fold(S::setzero_ps(), |x, y| S::add_ps(x, y)),
            )
        }
    );
    #[test]
    fn horizontal_add_test() {
        use rand::distributions::Standard;
        use rand::prelude::*;
        let x: Vec<f32> = thread_rng().sample_iter(&Standard).take(4000).collect();

        unsafe {
            let avx2_res = sum_simdeez_horizontal_avx2(&x);            
            let sse41_res = sum_simdeez_horizontal_sse41(&x);
            let sse_res = sum_simdeez_horizontal_sse2(&x);
            let scalar_res = sum_simdeez_horizontal_scalar(&x);
            assert_delta!(avx2_res, sse41_res, 0.01);            
            assert_delta!(sse_res, sse41_res, 0.01);
            assert_delta!(sse_res, scalar_res, 0.01);
        }
    }
    simd_runtime_generate!(
        fn sum_simdeez_horizontal_pd(x: &[f64]) -> f64 {
            assert!(x.len() % S::VF64_WIDTH == 0);

            S::horizontal_add_pd(
                x.chunks_exact(S::VF64_WIDTH)
                    .map(|y| S::loadu_pd(&y[0]))
                    .fold(S::setzero_pd(), |x, y| S::add_pd(x, y)),
            )
        }
    );
    #[test]
    fn horizontal_add_test_pd() {
        use rand::distributions::Standard;
        use rand::prelude::*;
        let x: Vec<f64> = thread_rng().sample_iter(&Standard).take(4000).collect();

        unsafe {
            let avx2_res = sum_simdeez_horizontal_pd_avx2(&x);            
            let sse41_res = sum_simdeez_horizontal_pd_sse41(&x);
            let sse_res = sum_simdeez_horizontal_pd_sse2(&x);
            let scalar_res = sum_simdeez_horizontal_pd_scalar(&x);            
            assert_delta!(avx2_res, sse41_res, 0.01);
            assert_delta!(sse_res, sse41_res, 0.01);
            assert_delta!(sse_res, scalar_res, 0.01);
        }
    }
    #[inline(always)]
    unsafe fn setlanetest<S: Simd>() -> f32 {
        let mut a = S::set1_ps(1.0);
        a[0] = 5.0;
        a[0]
    }
    unsafe fn setlanetest_avx2() -> f32 {
        setlanetest::<Avx2>()
    }
   
    #[inline(always)]
    unsafe fn gathertest_simd<S: Simd>() -> f32 {
        let a = [4.0, 3.0, 2.0, 1.0];
        let iarr = [0, 1, 2, 3];

        let index = S::loadu_epi32(&iarr[0]);
        let result = S::i32gather_ps(&a, index);
        result[0]
    }
    unsafe fn gathertest_sse2() -> f32 {
        gathertest_simd::<Sse2>()
    }

    #[inline(always)]
    unsafe fn overload_test<S: Simd>() -> i32 {
        let a = S::set1_epi32(3);
        let b = S::set1_epi32(2);
        let c = a + b; // 5
        let d = c * b; // 10
        let mut e = d - a; // 7
        e *= b; // 14
        let mut result = S::set1_epi32(9);
        result[0] = e[0];
        result[0]
    }
    unsafe fn overload_test_sse2() -> i32 {
        overload_test::<Sse2>()
    }

    #[test]
    fn overloads() {
        unsafe {
            assert_eq!(overload_test_sse2(), 14);
        }
    }
    #[inline(always)]
    unsafe fn overload_float_test<S: Simd>() -> f32 {
        let a = S::set1_ps(3.0);
        let b = S::set1_ps(2.0);
        let c = a + b; // 5
        let d = c * b; // 10
        let e = d - a; // 7
        let e = e / b; // 3.5
        let e = e * S::set1_ps(2.0); //7
        e[0]
    }
    unsafe fn overload_float_test_sse2() -> f32 {
        overload_float_test::<Sse2>()
    }

    #[test]
    fn overloads_float() {
        unsafe {
            assert_eq!(overload_float_test_sse2(), 7.0);
        }
    }
    #[test]
    fn setlane() {
        unsafe {
            assert_eq!(setlanetest_avx2(), 5.0);
        }
    }
    #[test]
    fn gathertest() {
        unsafe {
            assert_eq!(gathertest_sse2(), 4.0);
        }
    }


    const SPECIAL_F32: &[f32] = &[0.0f32, -0.0f32, f32::INFINITY, f32::NEG_INFINITY, f32::MIN, f32::MAX, 1.0, f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 1.844674e-25];
    //const SPECIAL_F32: &[f32] = &[f32::INFINITY, f32::NEG_INFINITY, f32::MIN, f32::MAX, -f32::MIN, -f32::MAX];

    struct FloatIter<S> {
      prelude: usize,
      v: f32,
      step: u32,
      phantom: std::marker::PhantomData<S>,
    }
    
    impl<S> FloatIter<S> {
      fn new(start: f32, step: u32) -> FloatIter<S> {
        FloatIter {
          prelude: 0,
          v: start, 
          step,
          phantom: std::marker::PhantomData,
        }
      }
    }
    
    impl<S: Simd> Iterator for FloatIter<S> {
      type Item = S::Vf32;
    
      fn next(&mut self) -> Option<Self::Item> {
    
        // test all the special values first
        if self.prelude < SPECIAL_F32.len() {
          let v = unsafe { S::set1_ps(SPECIAL_F32[self.prelude]) };
          self.prelude += 1;
          return Some(v);
        }
    
        if self.v == f32::INFINITY {
          return None;
        }
    
        let mut s = unsafe { S::set1_ps(0.0)};
    
        // Now walk through all the normal values
        for i in 0 .. S::VF32_WIDTH {
          let c = self.v;
          s[i] = self.v;
    
          for i in 0..self.step {
            self.v = self.v.next();
          }
        }
    
        Some(unsafe {
            S::loadu_ps(&s[0])
        })
      }
    }
    
    fn compare_simd_scalar<S: Simd, F1, F2>(f1: F1, f2: F2) where
      F1: Fn(S::Vf32) -> S::Vf32,
      F2: Fn(f32) -> f32,
    {
    
      let f32_iter = FloatIter::<S>::new(1e-36, 8);
    
      for arg in f32_iter {
    
        let result_simd = f1(arg);
    
        for i in 0 .. S::VF32_WIDTH {
    
          let result_scalar = f2(arg[i]);
          let diff = result_scalar.ulps(&result_simd[i]);
    
          if result_scalar.is_nan() && result_simd[i].is_nan() {
              continue;
          }

          if diff > 4 || diff < -4 {
            panic!("Arg: {}, SIMD: {}, Scalar: {}, ULPS: {}", arg[i], result_simd[i], result_scalar, diff);
            return;
          }
        }
      }
    }

    #[test]
    fn test_log() {
    
        fn log(v: <Avx2 as Simd>::Vf32) -> <Avx2 as Simd>::Vf32 {
            unsafe { Avx2::logf_ps(v)}
        }

        compare_simd_scalar::<Avx2,_,_>(log, f32::ln);
    }

    #[test]
    fn test_lgammaf_ps() {
    
        fn lgammaf(v: <Avx2 as Simd>::Vf32) -> <Avx2 as Simd>::Vf32 {
            unsafe { Avx2::lgammaf_ps(v)}
        }

        compare_simd_scalar::<Avx2,_,_>(lgammaf, ::libm::lgammaf);
    }
}
