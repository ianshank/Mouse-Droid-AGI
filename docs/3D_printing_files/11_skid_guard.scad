// ============================================================
//  MSE-6 — Corner Skid Guard  (print 4×)
//  Protects the 8mm shell overhang lips at each corner.
//  Snaps/glues onto the shell bottom edge corners.
//
//  Print in TPU 95A for best impact absorption,
//  or PETG if TPU not available.
//
//  Shell overhangs rover by 8mm each side.
//  Shell bottom edge at Z=10mm (3mm skirt below main body).
//
//  The guard wraps around the corner: 30mm each leg,
//  3mm thick, 12mm tall. Gentle 45° chamfer on bottom.
// ============================================================
$fn = 32;

// Guard dimensions
leg      = 30.0;   // length of each leg
guard_t  =  3.5;   // wall thickness
guard_h  = 12.0;   // height

// Inner radius matches shell corner radius (~8mm based on geometry)
inner_r  =  8.0;
outer_r  = inner_r + guard_t;

// Bottom chamfer
cham     =  3.0;

// Mounting: 2 small tabs with Ø2mm screw holes for M2 self-tapper
tab_l    =  8.0;
tab_w    =  4.0;
tab_t    =  2.0;
m2_d     =  2.2;

module skid_guard() {
    difference() {
        union() {
            // Corner arc
            difference() {
                cylinder(r = outer_r, h = guard_h);
                translate([0, 0, -0.1])
                    cylinder(r = inner_r, h = guard_h + 0.2);
                // Remove 3/4 leaving just the corner quadrant
                translate([-outer_r - 0.1, -outer_r - 0.1, -0.1])
                    cube([outer_r + 0.1, outer_r*2 + 0.2, guard_h + 0.2]);
                translate([-outer_r - 0.1, -outer_r - 0.1, -0.1])
                    cube([outer_r*2 + 0.2, outer_r + 0.1, guard_h + 0.2]);
            }
            // Left leg
            translate([0, inner_r, 0])
                cube([leg, guard_t, guard_h]);
            // Right leg
            translate([inner_r, 0, 0])
                cube([guard_t, leg, guard_h]);
        }

        // Bottom chamfer (all faces)
        translate([0, 0, -0.1])
            rotate([0, 0, 0])
                // Cut a chamfer bevel at bottom
                translate([-outer_r - 0.1, -outer_r - 0.1, 0])
                    cube([outer_r*2 + leg + 0.2, outer_r*2 + leg + 0.2, cham]);

        // Bevel the bottom edge properly
        for (angle = [0:5:90])
            rotate([0, 0, angle])
                translate([inner_r - 0.1, -guard_t - 0.1, 0])
                    rotate([0, 45, 0])
                        cube([cham*1.5, guard_t + 0.2, cham*1.5]);
    }

    // Mounting tabs (on inside face of legs)
    translate([leg*0.55, inner_r, guard_h - tab_t])
        difference() {
            cube([tab_l, guard_t, tab_t + 2]);
            translate([tab_l/2, -0.1, tab_t/2 + 1])
                rotate([-90, 0, 0])
                    cylinder(d = m2_d, h = guard_t + 0.2);
        }
    translate([inner_r, leg*0.55, guard_h - tab_t])
        difference() {
            cube([guard_t, tab_l, tab_t + 2]);
            translate([-0.1, tab_l/2, tab_t/2 + 1])
                rotate([90, 0, 90])
                    cylinder(d = m2_d, h = guard_t + 0.2);
        }
}

skid_guard();
