// ============================================================
//  MSE-6 — Top Dome Vent Grille Insert
//  Press-fit into a rectangular cutout in the top dome.
//  Allows Jetson fan heat to escape upward through the dome.
//
//  Suggested cutout location in top shell:
//    Near the top of the dome, centered (cosmetically hidden
//    in the raised central section of the MSE-6 top).
//  Cutout size: 50 × 20mm
//  Grille style: parallel horizontal slots (screen-accurate
//    to MSE-6 droid vent panels)
// ============================================================
$fn = 32;

// Cutout in top shell dome
cut_w      = 50.0;
cut_h      = 20.0;
dome_t     =  3.0;   // estimated dome wall thickness

// Grille body
body_w     = cut_w - 0.4;
body_h     = cut_h - 0.4;
body_d     = dome_t;

// Flange (inside dome)
flange_ow  =  3.5;
flange_t   =  2.0;

// Slot parameters
slot_h     =  2.5;   // slot height (open area)
bar_h      =  1.5;   // bar height between slots
bar_t      =  1.2;   // bar thickness (depth)
n_slots    = floor(body_h / (slot_h + bar_h));
margin_y   = (body_h - n_slots*(slot_h+bar_h) + bar_h) / 2;

// End bars
end_bar    =  2.0;

module vent_grille() {
    difference() {
        union() {
            // Body
            cube([body_w, body_d, body_h]);
            // Interior flange
            translate([-flange_ow, body_d, -flange_ow])
                cube([body_w + flange_ow*2, flange_t,
                      body_h + flange_ow*2]);
        }

        // Vent slots — horizontal, full width minus end bars
        for (i = [0 : n_slots-1]) {
            sz = margin_y + i*(slot_h + bar_h);
            translate([end_bar, -0.1, sz])
                cube([body_w - end_bar*2, body_d + flange_t + 0.2, slot_h]);
        }
    }
}

vent_grille();
