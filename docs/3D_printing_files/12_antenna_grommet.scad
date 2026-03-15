// ============================================================
//  MSE-6 — Antenna Pass-Through Grommet  (print 1-2×)
//  Press-fit plug for a Ø8mm hole in the shell.
//  Allows WiFi/BT antenna cable (typically SMA pigtail,
//  ~6mm OD) to exit the shell cleanly with strain relief.
//
//  Install: drill Ø8mm hole in convenient shell location,
//           press grommet in, route antenna cable through centre.
// ============================================================
$fn = 48;

// Shell hole
hole_d   =  8.0;
wall_t   =  4.0;

// Grommet body
body_d   = hole_d - 0.3;   // slight undersize for press fit
body_h   = wall_t;

// Flange
flange_d = hole_d + 6.0;
flange_t =  2.0;

// Cable bore (SMA pigtail ~6mm OD + 0.5mm)
cable_d  =  6.5;

// Strain relief collar (on inner side)
collar_d =  cable_d + 4.0;
collar_h =  6.0;
// Slit in collar creates flex fingers for grip
slit_w   =  1.2;

module grommet() {
    difference() {
        union() {
            // Press-fit body
            cylinder(d = body_d, h = body_h);
            // Exterior flange
            cylinder(d = flange_d, h = flange_t);
            // Interior strain-relief collar
            translate([0, 0, body_h])
                cylinder(d = collar_d, h = collar_h);
        }

        // Cable bore
        translate([0, 0, -0.1])
            cylinder(d = cable_d, h = body_h + collar_h + flange_t + 0.2);

        // 4 flex slits in collar
        for (a = [0, 90, 180, 270])
            rotate([0, 0, a])
                translate([-slit_w/2, -collar_d/2 - 0.1, body_h])
                    cube([slit_w, collar_d/2 + 0.1, collar_h]);

        // Countersink / chamfer on cable entry
        translate([0, 0, -0.1])
            cylinder(d1 = cable_d + 3, d2 = cable_d,
                     h = 1.5);
        translate([0, 0, body_h + collar_h - 1.5])
            cylinder(d1 = cable_d, d2 = cable_d + 3,
                     h = 1.6);
    }
}

grommet();
