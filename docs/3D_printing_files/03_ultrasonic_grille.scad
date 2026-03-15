// ============================================================
//  MSE-6 — Ultrasonic Aperture Grille
//  Press-fit insert for shell front wall.
//  Two circular windows align with HC-SR04 transducer cans.
//  Horizontal louvres block light/debris while passing sound.
//
//  HC-SR04 cans: Ø16mm, 26mm centre-to-centre
//  Sensor position on deck: roughly centred X on shell front wall
//  Shell front wall: X=81–127 (46mm wide), Z=10–43 (33mm tall)
//
//  Install: press into a 44×22mm rectangular cutout in the shell
//           front wall. Flange stops flush with outer face.
// ============================================================
$fn = 48;

// Aperture cutout in shell wall
aperture_w  = 44.0;    // wide enough for both cans + surround
aperture_h  = 22.0;    // tall enough for can OD + margin
wall_t      =  4.0;    // shell wall thickness

// Press-fit body
body_w      = aperture_w - 0.3;
body_h      = aperture_h - 0.3;
body_d      = wall_t;

// Interior flange
flange_ow   =  3.0;    // flange overhang each side
flange_t    =  2.0;

// HC-SR04 can windows
can_d       = 17.0;    // window diameter (Ø16mm can + 1mm clearance)
can_spacing = 26.0;    // centre-to-centre
can_cx      = body_w / 2;   // centred on grille
can_cy      = body_h / 2;

// Louvre parameters
louv_n      = 5;       // number of louvre bars across each window
louv_t      = 0.8;     // bar thickness (printable at 0.4mm nozzle)
louv_gap    = (can_d - louv_n * louv_t) / (louv_n + 1);

module can_window(cx, cy) {
    // Clear cylinder
    translate([cx, -0.1, cy])
        rotate([-90, 0, 0])
            cylinder(d = can_d, h = body_d + flange_t + 0.2);
}

module louvre_mask(cx, cy) {
    // Subtract to leave bars — actually ADD bars over the window
    // We build bars as thin rectangular strips across the window circle
    for (i = [0 : louv_n-1]) {
        bar_z = cy - can_d/2 + louv_gap*(i+1) + louv_t*i;
        translate([cx - can_d/2, -0.05, bar_z])
            cube([can_d, body_d + 0.1, louv_t]);
    }
}

module grille() {
    difference() {
        union() {
            // Press-fit body
            cube([body_w, body_d, body_h]);
            // Interior flange
            translate([-flange_ow, body_d, -flange_ow])
                cube([body_w + flange_ow*2, flange_t,
                      body_h + flange_ow*2]);
        }

        // Two can windows
        can_window(can_cx - can_spacing/2, can_cy);
        can_window(can_cx + can_spacing/2, can_cy);
    }

    // Add louvre bars back inside the windows
    intersection() {
        union() {
            louvre_mask(can_cx - can_spacing/2, can_cy);
            louvre_mask(can_cx + can_spacing/2, can_cy);
        }
        cube([body_w, body_d, body_h]);
    }
}

grille();
