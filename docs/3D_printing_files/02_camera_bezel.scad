// ============================================================
//  MSE-6 — Camera Viewport Bezel
//  Press-fit insert for the bottom shell front wall.
//  Provides a clear aperture for the IMX500 lens while
//  keeping the shell aesthetically closed.
//
//  Shell front wall: X=81–127mm (46mm wide), Z=10–43mm (33mm tall)
//  Wall thickness: ~4mm
//  Camera lens OD: 12.4mm (IMX500 dome)
//
//  Install: push into a 20×20mm square cutout in the shell
//           front wall, centred at X=104, Z=26 (mid-wall).
//           Glue or press-fit. Lens sits flush with outer face.
// ============================================================
$fn = 64;

// Shell wall aperture dimensions
aperture_w   = 20.0;   // square cutout width
aperture_h   = 20.0;   // square cutout height
wall_t       =  4.0;   // shell wall thickness (bezel depth)
flange_w     =  4.0;   // flange overhang on interior side
flange_t     =  2.0;   // flange thickness (stops bezel going too deep)

// Lens window
lens_d       = 13.5;   // clear aperture (IMX500 dome 12.4mm + 1.1mm clearance)
// Decorative outer ring
outer_d      = 17.0;   // chamfered outer ring OD on shell exterior face

// Press-fit body
body_w       = aperture_w - 0.3;  // slight undersize for press fit
body_h       = aperture_h - 0.3;
body_d       = wall_t;             // fills full wall thickness

module bezel() {
    difference() {
        union() {
            // Press-fit body
            cube([body_w, body_d, body_h]);

            // Interior flange (stops body from passing through)
            translate([-flange_w, body_d, -flange_w])
                cube([body_w + flange_w*2,
                      flange_t,
                      body_h + flange_w*2]);
        }

        // Lens aperture (centred in body)
        translate([body_w/2, -0.1, body_h/2])
            rotate([-90, 0, 0])
                cylinder(d = lens_d, h = body_d + flange_t + 0.2);

        // Chamfer on lens aperture exterior face
        translate([body_w/2, -0.1, body_h/2])
            rotate([-90, 0, 0])
                cylinder(d1 = lens_d + 3, d2 = lens_d,
                         h = 1.5);

        // Decorative chamfer ring on interior face
        translate([body_w/2, body_d + flange_t + 0.1, body_h/2])
            rotate([90, 0, 0])
                cylinder(d1 = outer_d + 2, d2 = outer_d,
                         h = 1.5);
    }
}

bezel();
