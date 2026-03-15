// ============================================================
//  MSE-6 — Deck Cable Penetration Plate
//  Snap-in insert for the cable hole in the internal deck (Z=42).
//  Protects CSI ribbon and HC-SR04 wires from deck edge damage.
//
//  The deck hole needs to be cut/drilled: ~30 × 14mm rectangle
//  centred at ~X=104, Y=91 (middle of the 156×80mm deck area).
//
//  Features:
//    – Press-fit body fills the deck hole
//    – Wide flange on top face of deck (stops it falling through)
//    – Two slots: wide (CSI ribbon 16mm) + narrow (4-wire HC-SR04)
//    – Chamfered slot edges to prevent ribbon chafing
//    – Snap tabs on underside
// ============================================================
$fn = 32;

// Deck hole (cut this in the shell deck)
hole_w   = 30.0;
hole_d   = 14.0;
deck_t   =  3.0;   // deck thickness (estimated)

// Insert body
body_w   = hole_w - 0.4;
body_d   = hole_d - 0.4;
body_h   = deck_t;

// Top flange
flange_ow =  3.0;
flange_t  =  2.5;

// CSI ribbon slot
csi_w    = 17.0;
csi_h    =  1.5;   // ribbon thickness + 0.5mm

// GPIO wire slot  
gpio_w   =  8.0;
gpio_h   =  4.0;

// Chamfer on slot edges
cham     =  1.0;

// Snap tab (underside)
tab_l    =  8.0;
tab_w    =  2.5;
tab_h    =  2.0;

module cable_plate() {
    difference() {
        union() {
            // Press-fit body
            cube([body_w, body_d, body_h]);
            // Top flange
            translate([-flange_ow, -flange_ow, body_h])
                cube([body_w + flange_ow*2, body_d + flange_ow*2, flange_t]);
            // Snap tabs on underside
            for (tx = [body_w*0.25, body_w*0.75])
                translate([tx - tab_w/2, body_d/2 - tab_l/2, -tab_h])
                    cube([tab_w, tab_l, tab_h + 0.1]);
        }

        // CSI ribbon slot (wide, thin) — centred
        translate([body_w/2 - csi_w/2, -0.1, body_h - csi_h])
            cube([csi_w, body_d + 0.2, csi_h + flange_t + 0.1]);

        // Chamfer ribbon slot edges
        translate([body_w/2 - csi_w/2 - cham, -0.1, body_h - csi_h - cham])
            rotate([0, 45, 0])
                cube([cham*1.4, body_d + 0.2, cham*1.4]);
        translate([body_w/2 + csi_w/2, -0.1, body_h - csi_h - cham])
            rotate([0, 45, 0])
                cube([cham*1.4, body_d + 0.2, cham*1.4]);

        // GPIO wire slot (narrower, taller)
        translate([body_w/2 - gpio_w/2, -0.1, -0.1])
            cube([gpio_w, body_d + 0.2, gpio_h + 0.1]);

        // Chamfer gpio slot
        translate([body_w/2 - gpio_w/2, body_d + 0.1, gpio_h])
            rotate([0, 0, 180]) rotate([45, 0, 0])
                cube([gpio_w, cham*1.4, cham*1.4]);

        // Snap tab bevel (makes them flex)
        for (tx = [body_w*0.25, body_w*0.75])
            translate([tx - tab_w/2 - 0.1, body_d/2 - tab_l/2 - 0.1, -tab_h - 0.1])
                rotate([0, 0, 0])
                    cube([tab_w + 0.2, 1.5, tab_h * 0.7]);
    }
}

cable_plate();
