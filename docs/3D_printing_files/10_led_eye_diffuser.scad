// ============================================================
//  MSE-6 — LED Eye Diffuser  (print 2×)
//  Screen-accurate rectangular "eye" lights for MSE-6 droid.
//  Press-fit into two slots cut in the front face of the
//  bottom shell.
//
//  MSE-6 eye proportions: ~14×8mm rectangular window
//  LED behind: any 5mm LED or small LED strip segment
//
//  Features:
//    – Frosted diffuser face (thin wall for light transmission)
//    – Interior LED holder pocket (5mm LED or 10mm strip)
//    – Flange for mounting in shell wall cutout
//    – Two variants: left eye, right eye (mirrored)
//
//  Shell cutout: 16 × 10mm per eye
//  Position: X≈88 and X≈116, Z≈35 (front wall, ~35mm height)
// ============================================================
$fn = 32;

MIRROR = false;  // set true for right eye

// Shell wall cutout
cut_w     = 16.0;
cut_h     = 10.0;
wall_t    =  4.0;

// Bezel body
body_w    = cut_w - 0.3;
body_h    = cut_h - 0.3;
body_d    = wall_t;

// Flange
flange_ow =  2.5;
flange_t  =  1.8;

// Diffuser face (thin for light transmission)
diff_t    =  0.8;   // thin wall — use clear/translucent filament

// LED pocket behind diffuser
led_w     =  8.0;   // 5mm LED fits, or strip segment
led_h     =  5.0;
led_d     =  6.0;   // pocket depth

// Decorative border on face (thicker surround)
border    =  1.5;

module led_eye() {
    mirror_v = MIRROR ? [1,0,0] : [0,0,0];
    mirror(mirror_v)
    difference() {
        union() {
            // Body
            cube([body_w, body_d, body_h]);
            // Flange (interior)
            translate([-flange_ow, body_d, -flange_ow])
                cube([body_w + flange_ow*2, flange_t,
                      body_h + flange_ow*2]);
        }

        // Diffuser window (thin front wall remains)
        translate([border, diff_t, border])
            cube([body_w - border*2,
                  body_d - diff_t + flange_t + 0.1,
                  body_h - border*2]);

        // LED pocket (blind hole from interior)
        translate([body_w/2 - led_w/2, body_d - led_d, body_h/2 - led_h/2])
            cube([led_w, led_d + 0.1, led_h]);

        // Corner chamfers on face
        chamf = 1.2;
        for (cx=[0, body_w], cz=[0, body_h])
            translate([cx, -0.1, cz])
            rotate([0, cx==0 ? -45 : 45, 0])
                cube([chamf*1.4, body_d + 0.2, chamf*1.4]);
    }
}

led_eye();
