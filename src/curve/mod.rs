pub mod edwards;
pub mod montgomery;
pub(crate) mod scalar_mul;
pub(crate) mod twedwards;

use crate::field::FieldElement;
pub use edwards::EdwardsPoint;
pub use montgomery::MontgomeryPoint;

use crate::curve::edwards::affine::AffinePoint;
use subtle::{ConditionallyNegatable, ConditionallySelectable, ConstantTimeEq};

pub(crate) fn map_to_curve_elligator2(u: &FieldElement) -> AffinePoint {
    let mut t1 = u.square(); // 1.   t1 = u^2
    t1 *= FieldElement::Z; // 2.   t1 = Z * t1              // Z * u^2
    let e1 = t1.ct_eq(&FieldElement::MINUS_ONE); // 3.   e1 = t1 == -1            // exceptional case: Z * u^2 == -1
    t1.conditional_assign(&FieldElement::ZERO, e1); // 4.   t1 = CMOV(t1, 0, e1)     // if t1 == -1, set t1 = 0
    let mut x1 = t1 + FieldElement::ONE; // 5.   x1 = t1 + 1
    x1 = x1.invert(); // 6.   x1 = inv0(x1)
    x1 *= -FieldElement::J; // 7.   x1 = -A * x1             // x1 = -A / (1 + Z * u^2)
    let mut gx1 = x1 + FieldElement::J; // 8.  gx1 = x1 + A
    gx1 *= x1; // 9.  gx1 = gx1 * x1
    gx1 += FieldElement::ONE; // 10. gx1 = gx1 + B
    gx1 *= x1; // 11. gx1 = gx1 * x1            // gx1 = x1^3 + A * x1^2 + B * x1
    let x2 = -x1 - FieldElement::J; // 12.  x2 = -x1 - A
    let gx2 = t1 * gx1; // 13. gx2 = t1 * gx1
    let e2 = gx1.is_square(); // 14.  e2 = is_square(gx1)
    let x = FieldElement::conditional_select(&x2, &x1, e2); // 15.   x = CMOV(x2, x1, e2)    // If is_square(gx1), x = x1, else x = x2
    let y2 = FieldElement::conditional_select(&gx2, &gx1, e2); // 16.  y2 = CMOV(gx2, gx1, e2)  // If is_square(gx1), y2 = gx1, else y2 = gx2
    let mut y = y2.sqrt(); // 17.   y = sqrt(y2)
    let e3 = y.is_negative(); // 18.  e3 = sgn0(y) == 1
    y.conditional_negate(e2 ^ e3); //       y = CMOV(-y, y, e2 xor e3)
    AffinePoint { x, y }
}

pub(crate) fn iso448(p: &AffinePoint) -> AffinePoint {
    let x = p.x;
    let y = p.y;
    let mut t0 = x.square(); // x^2
    let t1 = t0 + FieldElement::ONE; // x^2+1
    t0 -= FieldElement::ONE; // x^2-1
    let mut t2 = y.square(); // y^2
    t2 = t2.double(); // 2y^2
    let t3 = x.double(); // 2x

    let mut t4 = t0 * y; // y(x^2-1)
    t4 = t4.double(); // 2y(x^2-1)
    let xNum = t4.double(); // xNum = 4y(x^2-1)

    let mut t5 = t0.square(); // x^4-2x^2+1
    t4 = t5 + t2; // x^4-2x^2+1+2y^2
    let xDen = t4 + t2; // xDen = x^4-2x^2+1+4y^2

    t5 *= x; // x^5-2x^3+x
    t4 = t2 * t3; // 4xy^2
    let yNum = t4 - t5; // yNum = -(x^5-2x^3+x-4xy^2)

    t4 = t1 * t2; // 2x^2y^2+2y^2
    let yDen = t5 - t4; // yDen = x^5-2x^3+x-2x^2y^2-2y^2

    AffinePoint {
        x: xNum * xDen.invert(),
        y: yNum * yDen.invert(),
    }
}
