use crate::Simd;
use libm::lgammaf;

const BIG_F32: f32 = ((1u64<<32) as f32) * ((1u64<<32) as f32); 
pub trait SimdSpecial: Simd {

    #[inline(always)]
    unsafe fn exponent_ps(a: Self::Vf32) -> Self::Vi32 {
        let mut v = Self::castps_epi32(a);
        v = v >> 23;
        v = v & Self::set1_epi32(0xff);
        v - Self::set1_epi32(0x7f)
    }  

    #[inline]
    unsafe fn fraction2_ps(a: Self::Vf32) -> Self::Vf32 {
        let t1 = Self::castps_epi32(a);
        let t2 = t1 & Self::set1_epi32(0x007FFFFF) | Self::set1_epi32(0x3F000000);
        Self::castepi32_ps(t2)
    }

    unsafe fn logf_ps(v: Self::Vf32) -> Self::Vf32 {

        let denormal_mask = Self::cmplt_ps(v, Self::set1_ps(f32::MIN_POSITIVE));
        let v_big = v * Self::set1_ps(BIG_F32);

        // v with denormals multipled by 2^64
        let d = Self::blendv_ps(v, v_big, denormal_mask);

        // extract exponent
        let e1 = Self::exponent_ps(d * Self::set1_ps(1.0 / 0.75));

        // extract mantissa by incrementing exponent of d by e
        let q = Self::castps_epi32(d);
        let m = Self::castepi32_ps(q + ((Self::set1_epi32(0) - e1) << 23));

        // update exponent for denormals
        let e = Self::blendv_epi32(e1, e1 - Self::set1_epi32(64), Self::castps_epi32(denormal_mask));

        // main numerical stuff
        let x = (Self::set1_ps(-1.0) + m) / ((Self::set1_ps(1.0) + m));
        let x2 = x * x;

        let mut t = Self::set1_ps(0.2392828464508056640625);
        t = Self::fmadd_ps(t, x2, Self::set1_ps(0.28518211841583251953125));
        t = Self::fmadd_ps(t, x2, Self::set1_ps(0.400005877017974853515625));
        t = Self::fmadd_ps(t, x2, Self::set1_ps(0.666666686534881591796875));
        t = Self::fmadd_ps(t, x2, Self::set1_ps(2.0));

        // add exponent to result
        let mut x = Self::fmadd_ps(x, t, Self::set1_ps(0.693147180559945286226764) * Self::cvtepi32_ps(e));

        // handle infinities in input
        x = Self::blendv_ps(x, Self::set1_ps(f32::INFINITY), Self::cmpeq_ps(d, Self::set1_ps(f32::INFINITY)));

        // if input is Nan or negative, set to Nan
        x = Self::blendv_ps(x, Self::set1_ps(f32::NAN),
            Self::or_ps(
                Self::cmplt_ps(d, Self::set1_ps(0.0)), 
                Self::cmpneq_ps(d, d))
        );

        // if input is +0, output is -INF
        x = Self::blendv_ps(x, Self::set1_ps(f32::NEG_INFINITY), Self::cmpeq_ps(v, Self::set1_ps(0.0)));
        return x;
    }

    /// Uses a SIMD path for 8.0 < x < 2.0**58, falls back to scalar with libm for
    /// any arguments outside that range.
    unsafe fn lgammaf_ps(x: Self::Vf32) -> Self::Vf32 {
    
        let mask1 = Self::cmpgt_ps(x, Self::set1_ps(UPPER));
        let mask2 = Self::cmplt_ps(x, Self::set1_ps(LOWER));
        let nan = Self::cmpneq_ps(x, x);
        let bad_vals = Self::or_ps(nan, Self::or_ps(mask1, mask2));

        /* 8.0 <= x < 2**58 */
        let t = Self::logf_ps(x);
        let z = Self::div_ps(Self::set1_ps(1.0), x);
        let y = Self::mul_ps(z, z);
    
        let mut w = Self::fmadd_ps(y, Self::set1_ps(W6), Self::set1_ps(W5));
        w = Self::fmadd_ps(w, y, Self::set1_ps(W4));
        w = Self::fmadd_ps(w, y, Self::set1_ps(W3));
        w = Self::fmadd_ps(w, y, Self::set1_ps(W2));
        w = Self::fmadd_ps(w, y, Self::set1_ps(W1));
    
        w = Self::fmadd_ps(w, z, Self::set1_ps(W0));
    
        let mut r = Self::fmadd_ps(
            Self::sub_ps(x, Self::set1_ps(0.5)),
            Self::sub_ps(t, Self::set1_ps(1.0)),
            w
        );

        // if any of the values are out of the range 8..2^58, fill them in manually using libm
        let all_in_range = Self::testz_ps(bad_vals, bad_vals);
        if all_in_range == 0 {

            for i in 0..Self::VF32_WIDTH {
                let bits = bad_vals[i].to_bits();
                if bits == 0 { continue; }

                r[i] = lgammaf(x[i]);
            }
        }

        r
    }
}


const W0: f32 = 4.18938533204672725052e-01; /* 0x3FDACFE3, 0x90C97D69 */
const W1: f32 = 8.33333333333329678849e-02; /* 0x3FB55555, 0x5555553B */
const W2: f32 = -2.77777777728775536470e-03; /* 0xBF66C16C, 0x16B02E5C */
const W3: f32 = 7.93650558643019558500e-04; /* 0x3F4A019F, 0x98CF38B6 */
const W4: f32 = -5.95187557450339963135e-04; /* 0xBF4380CB, 0x8C0FE741 */
const W5: f32 = 8.36339918996282139126e-04; /* 0x3F4B67BA, 0x4CDAD5D1 */
const W6: f32 = -1.63092934096575273989e-03; /* 0xBF5AB89D, 0x0B9E43E4 */

const UPPER: f32 = 288230376151711744.0;
const LOWER: f32 = 8.0;


impl SimdSpecial for crate::avx2::Avx2 {

}
