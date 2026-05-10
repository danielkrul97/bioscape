use wide::f32x8;

/// Padé(3,2) tanh approximation, clamped to ±3. At |x|=3 numerator and
/// denominator both equal 108, so the function saturates cleanly to ±1.
/// Max error ~2 % inside the active range — not meaningful for tanh
/// activation (signal is already saturating there).
///
/// The scalar and SIMD forms share the exact same expression so that
/// `Cppn::forward` and `Cppn::forward_batch_x8` produce bit-identical
/// outputs when `Brain::from_cppn` mixes scalar and SIMD paths to handle
/// trailing partial batches.
#[inline]
pub fn tanh_fast_scalar(x: f32) -> f32 {
    let cx = x.clamp(-3.0, 3.0);
    let x2 = cx * cx;
    cx * (27.0 + x2) / (27.0 + 9.0 * x2)
}

#[inline]
pub fn tanh_fast_simd(x: f32x8) -> f32x8 {
    let x = x.fast_max(f32x8::splat(-3.0)).fast_min(f32x8::splat(3.0));
    let x2 = x * x;
    x * (f32x8::splat(27.0) + x2) / (f32x8::splat(27.0) + f32x8::splat(9.0) * x2)
}
