// ============================================================
//  MSE-6 — Service Access Panel + Frame
//  Two parts: Frame (glues into shell side wall cutout)
//             Panel (removable, 2× M3 captive screws)
//
//  Location: Right side wall of bottom shell (~X=170, Z=20–55)
//  Gives access to: Jetson USB-A port, SD card, battery swap
//
//  Cutout in shell: 52 × 32mm
//  Frame: press-fits into cutout, provides M3 boss for screws
//  Panel: snaps onto frame with 2× M3×8 screws (recessed heads)
// ============================================================
$fn = 48;

PART = "both"; // "frame", "panel", or "both"

// Cutout dimensions
cut_w    = 52.0;
cut_h    = 32.0;
wall_t   =  4.0;   // shell wall thickness

// Frame
fr_ow    =  4.0;   // overhang each side
fr_t     =  2.0;   // flange thickness
body_w   = cut_w - 0.4;
body_h   = cut_h - 0.4;
m3_d     =  3.4;   // clearance
m3_boss  =  7.0;   // boss OD
boss_h   = wall_t + fr_t + 3.0;  // enough for screw engagement

// Panel
panel_t  =  2.5;
inset    =  0.4;   // panel slightly smaller than frame opening

module frame() {
    difference() {
        union() {
            // Press-fit body (into shell wall)
            cube([body_w, wall_t, body_h]);
            // Inner flange (interior side)
            translate([-fr_ow, wall_t, -fr_ow])
                cube([body_w + fr_ow*2, fr_t,
                      body_h + fr_ow*2]);
            // M3 screw bosses (2 corners, interior flange)
            for (bx = [fr_ow*0.8, body_w - fr_ow*0.8],
                 bz = [cut_h*0.18, cut_h*0.82])
                translate([bx, wall_t, bz - body_h/2 + body_h/2])
                    rotate([90,0,0])
                        translate([0, 0, -fr_t - 0.1])
                            cylinder(d = m3_boss, h = boss_h + fr_t);
        }
        // Clear opening
        translate([0, -0.1, 0])
            cube([body_w, wall_t + 0.2, body_h]);
        // M3 screw holes through bosses
        for (bx = [fr_ow*0.8, body_w - fr_ow*0.8],
             bz = [cut_h*0.18, cut_h*0.82])
            translate([bx, wall_t + boss_h, bz - body_h/2 + body_h/2])
                rotate([90,0,0])
                    cylinder(d = m3_d, h = boss_h + fr_t + 0.2);
    }
}

module panel() {
    pw = body_w - inset*2 + fr_ow*2;
    ph = body_h - inset*2 + fr_ow*2;
    difference() {
        cube([pw, panel_t, ph]);
        // M3 countersunk holes
        for (bx = [fr_ow*0.8, pw - fr_ow*0.8],
             bz = [cut_h*0.18 + fr_ow, cut_h*0.82 + fr_ow])
            translate([bx + inset - fr_ow*0.1, -0.1, bz])
                rotate([-90,0,0]) {
                    cylinder(d = m3_d, h = panel_t + 0.2);
                    cylinder(d = 6.0, h = 1.8); // countersink
                }
        // Finger notch to pull panel
        translate([pw/2 - 8, -0.1, ph*0.48])
            cube([16, panel_t + 0.2, 5]);
    }
}

if (PART == "frame" || PART == "both")  frame();
if (PART == "panel" || PART == "both")
    translate([0, wall_t + 15, 0]) panel();
