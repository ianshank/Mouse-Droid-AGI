// ============================================================
//  MSE-6 — Shell Alignment Pin Insert  (print 4×)
//  Bonds into the bottom shell top edge (Z=74.2mm mating face)
//  to register the top dome.  Print 2 with pin UP (bottom shell)
//  and 2 with socket cavity (top shell).
//
//  Set pin_mode = true  → pin   (glues into bottom shell rim)
//  Set pin_mode = false → socket (glues into top shell rim)
//
//  Also includes: M3 screw boss version for the 2 side positions
//  (set screw_mode = true for screw boss variant)
// ============================================================
$fn = 48;

// ── Toggle these ────────────────────────────────────────────
pin_mode   = true;   // true=pin, false=socket
screw_mode = false;  // true=M3 screw boss instead of dowel pin

// ── Dimensions ──────────────────────────────────────────────
// Body that glues into shell rim
body_d     = 10.0;  // outer diameter of insert body
body_h     =  8.0;  // depth that sits in shell rim pocket
flange_d   = 14.0;  // flange that rests on shell mating face
flange_t   =  2.0;

// Dowel pin
pin_d      =  4.0;  // 4mm dowel pin OD
pin_h      = 10.0;  // pin protrusion above flange
pin_hole_d =  4.2;  // socket hole (0.2mm clearance)
pin_hole_h = 10.5;  // socket depth

// M3 screw boss
m3_boss_d  =  7.0;
m3_d       =  3.2;  // M3 self-tap
m3_h       = 12.0;  // screw engagement depth

module alignment_pin() {
    difference() {
        union() {
            // Insert body (goes into shell rim)
            cylinder(d = body_d, h = body_h);
            // Flange
            translate([0, 0, body_h])
                cylinder(d = flange_d, h = flange_t);
            // Pin or screw boss above flange
            translate([0, 0, body_h + flange_t]) {
                if (screw_mode) {
                    cylinder(d = m3_boss_d, h = m3_h);
                } else if (pin_mode) {
                    cylinder(d = pin_d, h = pin_h);
                }
            }
        }
        // Socket hole (if socket mode)
        if (!pin_mode && !screw_mode) {
            translate([0, 0, body_h + flange_t - 0.1])
                cylinder(d = pin_hole_d, h = pin_hole_h);
            translate([0, 0, -0.1])
                cylinder(d = pin_hole_d, h = body_h + flange_t + pin_hole_h + 0.2);
        }
        // M3 hole through screw boss
        if (screw_mode) {
            translate([0, 0, -0.1])
                cylinder(d = m3_d, h = body_h + flange_t + m3_h + 0.2);
        }
        // Hollow body core (reduce material, keep strength)
        translate([0, 0, -0.1])
            cylinder(d = body_d - 3.0, h = body_h - 1);
    }
}

alignment_pin();
