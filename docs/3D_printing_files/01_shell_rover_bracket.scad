// ============================================================
//  MSE-6 — Shell-to-Rover Lock Bracket  (print 4×)
//  Clamps the MSE-6 printed shell onto the Wave Rover
//  aluminum expansion plate so it can't shift or rattle.
//
//  How it works:
//    – Vertical leg presses against inside of shell skirt wall
//    – Horizontal foot sits on Wave Rover expansion plate
//    – M3 screw through foot into one of the rover plate M3 holes
//    – Top tab hooks over the shell skirt lip (Z=13mm)
//    – Print 4, place at ~X=40, X=170 (front+rear both sides)
// ============================================================
$fn = 48;

// Rover plate sits at Z=0 in this part's local frame
// Shell skirt inner wall is ~4mm thick, skirt height 3mm (Z=10→13)

foot_l    = 28;    // length along rover plate
foot_w    = 14;    // width
foot_t    =  3;    // thickness of foot plate
leg_h     = 33;    // height of vertical leg (rover plate to shell skirt base ≈ 33mm)
                   // shell sits on rover plate; skirt bottom at rover surface
leg_t     =  3;    // leg thickness
tab_h     =  4;    // hook tab height over skirt lip
tab_d     =  5;    // hook tab depth (grips skirt lip)
tab_gap   =  3.5;  // clearance slot = skirt thickness (3mm) + 0.5mm
m3_d      =  3.4;  // M3 clearance hole through foot
m3_csink  =  6.0;  // M3 countersink diameter
csink_d   =  2.0;  // countersink depth
fillet    =  2.5;

module bracket() {
    difference() {
        union() {
            // Horizontal foot
            cube([foot_l, foot_w, foot_t]);

            // Vertical leg (rear of foot)
            translate([0, foot_w - leg_t, foot_t])
                cube([foot_l, leg_t, leg_h]);

            // Hook tab at top of leg (clips over shell skirt lip)
            translate([0, foot_w - leg_t - tab_d, foot_t + leg_h])
                cube([foot_l, leg_t + tab_d, tab_gap + leg_t]);

            // Fillet between foot and leg
            translate([0, foot_w - leg_t - fillet, foot_t])
                rotate([0, 90, 0])
                    cylinder(r = fillet, h = foot_l);
        }

        // M3 hole through foot (2 holes)
        for (xi = [foot_l*0.25, foot_l*0.75])
            translate([xi, foot_w/2, -0.1]) {
                cylinder(d = m3_d, h = foot_t + 0.2);
                cylinder(d = m3_csink, h = csink_d + 0.1);  // countersink
            }

        // Slot in hook tab = shell skirt thickness + clearance
        translate([-0.1, foot_w - leg_t - tab_d - 0.1,
                   foot_t + leg_h + tab_gap])
            cube([foot_l + 0.2, tab_d + leg_t + 0.2, leg_t + 0.2]);

        // Lighten leg with slot
        translate([foot_l*0.2, foot_w - leg_t - 0.1, foot_t + 8])
            cube([foot_l*0.6, leg_t + 0.2, leg_h - 16]);
    }
}

bracket();
