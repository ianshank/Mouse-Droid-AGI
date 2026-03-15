// ============================================================
//  MSE-6 Mouse Droid — Raspberry Pi AI Camera (IMX500) Mount
//  Fits mse6_v6_bottom.stl (210 x 184 x 64 mm chassis)
//  v2 – manifold-clean
// ============================================================

$fn = 64;

// ── Parameters ──────────────────────────────────────────────
cam_w          = 25.0;
cam_d          = 24.0;
cam_h          = 1.6;
cam_lens_d     = 12.4;
cam_lens_x     = 12.5;
cam_lens_y     = 12.0;
cam_m2_pitch_x = 21.0;
cam_m2_pitch_y = 12.5;
cam_m2_d       = 2.2;
cam_m2_cx      = 2.0;
cam_m2_cy      = 5.75;

wall           = 2.0;
floor_t        = 2.0;
retention_h    = 3.0;

tilt_angle     = 10;

base_w         = 36.0;
base_d         = 20.0;
base_t         = 3.0;

m3_d           = 3.4;
// Y-axis spacing between M3 boss holes on chassis
m3_ear_y_span  = 24.0;   // ear spacing within mount (narrower than 72mm boss span)
                           // Set mount at chassis center; use boss holes at 54/126 Y

cable_w        = 6.0;
cable_h        = 3.5;

// ── Camera cradle ────────────────────────────────────────────
module camera_cradle() {
    difference() {
        translate([-wall, -wall, 0])
            cube([cam_w + wall*2, cam_d + wall*2,
                  floor_t + cam_h + retention_h]);

        // PCB pocket
        translate([0, 0, floor_t])
            cube([cam_w + 0.3, cam_d + 0.3,
                  cam_h + retention_h + 0.1]);

        // Lens window (large enough for IMX500 dome)
        translate([cam_lens_x, cam_lens_y, -0.1])
            cylinder(d = cam_lens_d + 1.0, h = floor_t + 2);

        // M2 access holes
        for (ix = [0 : 1 : 1], iy = [0 : 1 : 1])
            translate([cam_m2_cx + ix * cam_m2_pitch_x,
                       cam_m2_cy + iy * cam_m2_pitch_y,
                       -0.1])
                cylinder(d = cam_m2_d,
                         h = floor_t + cam_h + retention_h + 1);

        // Cable exit slot at back wall
        translate([cam_w/2 - cable_w/2,
                   cam_d + wall - 0.1,
                   floor_t + cam_h])
            cube([cable_w, wall + 1, cable_h]);
    }
}

// ── Base plate with integral M3 ears ─────────────────────────
module base_plate() {
    difference() {
        union() {
            // Main base
            cube([base_w, base_d, base_t]);

            // Front ear (Y = 0 side)
            translate([base_w/2, -m3_ear_y_span/2 + base_d/2, 0])
                cylinder(r = 5, h = base_t);

            // Rear ear (Y = base_d side)
            translate([base_w/2, base_d + m3_ear_y_span/2 - base_d/2, 0])
                cylinder(r = 5, h = base_t);
        }

        // M3 holes in ears
        translate([base_w/2, -m3_ear_y_span/2 + base_d/2, -0.1])
            cylinder(d = m3_d, h = base_t + 1);
        translate([base_w/2, base_d + m3_ear_y_span/2 - base_d/2, -0.1])
            cylinder(d = m3_d, h = base_t + 1);

        // Optional flush M3 holes through base body
        translate([base_w/2, base_d * 0.3, -0.1])
            cylinder(d = m3_d, h = base_t + 1);
        translate([base_w/2, base_d * 0.7, -0.1])
            cylinder(d = m3_d, h = base_t + 1);
    }
}

// ── Assembly ─────────────────────────────────────────────────
module camera_mount() {
    union() {
        base_plate();

        // Cradle sits on top of base, tilted forward
        translate([(base_w - cam_w) / 2,
                   (base_d - cam_d) / 2,
                   base_t])
        rotate([-tilt_angle, 0, 0])
            camera_cradle();
    }
}

camera_mount();
