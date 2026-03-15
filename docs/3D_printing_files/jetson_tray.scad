// ============================================================
//  MSE-6 Mouse Droid — Jetson Orin Nano Devkit Tray
//  Mounts Jetson UNDER the MSE-6 shell, directly on the
//  Wave Rover expansion plate (aluminum top plate).
//
//  Stack from Wave Rover plate upward:
//    3.0 mm  — tray base
//    5.0 mm  — M3 brass standoffs (Jetson sits on these)
//   21.0 mm  — Jetson Orin Nano devkit (incl. heatsink/fan)
//   ──────────
//   29.0 mm  total  (32 mm available under MSE-6 deck → 3 mm margin)
//
//  Tray → Wave Rover plate:
//    4× M3 clearance holes matching Wave Rover expansion plate
//    pattern: 96.744 mm × ~38 mm C-C (from official DXF)
//    M3 screws from below (countersunk or button head into rover plate)
//
//  Tray → Jetson:
//    4× M3 standoff bosses, 86 mm × 57 mm C-C
//    (Jetson Orin Nano devkit hole pattern from carrier board spec)
//    Use M3×5 brass standoffs + M3×6 screws into Jetson holes
//
//  Cable management:
//    - CSI ribbon slot (rear center): routes up through MSE-6 deck hole
//    - GPIO/HC-SR04 wire channel (rear left)
//    - DC barrel jack notch (rear right — 5.5×2.5mm jack)
//    - USB-A notch (front right — for debug/mouse/keyboard)
//
//  Print: PETG recommended (heat near Jetson), 0.2mm layers, 4 walls
//  Hardware: 4× M3×5 brass standoffs (press-fit or thread into bosses)
// ============================================================

$fn = 64;

// ── Rover plate M3 hole pattern (from DXF, normalized) ──────
// 6 holes total; we use the 4 that give a stable rectangular pattern
// Holes 1,2,5,6 form ~96.7 × 40mm rectangle
rover_m3 = [
    [  0.000,  0.000 ],   // ← hole 1 (left front)
    [  0.000, 40.000 ],   // ← hole 2 (left rear)
    [ 96.744,  1.940 ],   // ← hole 5 (right front)
    [ 96.744, 38.060 ],   // ← hole 6 (right rear)
];

// ── Jetson Orin Nano hole pattern ────────────────────────────
// 4× M3, 86mm × 57mm C-C, offset 7mm from board X edge, 11mm from Y edge
jetson_hole_xspan = 86.0;
jetson_hole_yspan = 57.0;
jetson_l          = 100.0;
jetson_w          = 79.0;

// ── Tray parameters ──────────────────────────────────────────
tray_l      = 114.0;   // slightly larger than Jetson (100mm) for ear overhangs
tray_w      =  88.0;   // slightly larger than Jetson (79mm)
tray_t      =   3.0;   // base plate thickness
wall_t      =   1.8;   // rim wall thickness (keeps Jetson located)
rim_h       =   2.0;   // rim height above base (just a registration lip)

// Standoff boss: raised cylinder that accepts M3×5 brass standoff
boss_r      =   4.0;   // boss outer radius
boss_h      =   4.0;   // boss height above tray base
boss_hole_d =   3.0;   // M3 tap diameter (tight for self-tapping or pre-tap)
                        // change to 3.4 if using clearance + nut below

// Rover M3 clearance holes through tray base
rover_hole_d = 3.4;    // M3 clearance

// Tray origin: bottom-left corner
// We centre the rover hole pattern on the tray
rover_origin_x = (tray_l - 96.744) / 2;
rover_origin_y = (tray_w - 40.000) / 2;

// Jetson sits centred on tray
jetson_origin_x = (tray_l - jetson_l) / 2;
jetson_origin_y = (tray_w - jetson_w) / 2;

// Jetson hole positions relative to tray origin
function jetson_hole(xi, yi) = [
    jetson_origin_x + (xi == 0 ? (jetson_l - jetson_hole_xspan)/2
                                : (jetson_l - jetson_hole_xspan)/2 + jetson_hole_xspan),
    jetson_origin_y + (yi == 0 ? (jetson_w - jetson_hole_yspan)/2
                                : (jetson_w - jetson_hole_yspan)/2 + jetson_hole_yspan)
];

// ── Cable / connector cutouts ────────────────────────────────
// Jetson rear edge (Y = jetson_origin_y + jetson_w)
// CSI ribbon connector: ~16mm wide, at board centre X
csi_slot_w   = 18;    // ribbon cable width + clearance
csi_slot_d   = 12;    // depth into tray wall
csi_slot_x   = tray_l/2 - csi_slot_w/2;
csi_rear_y   = jetson_origin_y + jetson_w - csi_slot_d;

// GPIO / HC-SR04 wires — left rear area, open channel
gpio_slot_w  = 12;
gpio_slot_x  = jetson_origin_x;

// DC barrel jack — right side of Jetson
// Jack is on the right short edge of the board (X max side)
barrel_w     = 10;    // 5.5mm OD jack + clearance
barrel_h     =  8;    // height of notch

// ── Modules ─────────────────────────────────────────────────

module boss(x, y) {
    translate([x, y, tray_t])
    difference() {
        cylinder(r = boss_r, h = boss_h);
        translate([0, 0, -0.1])
            cylinder(d = boss_hole_d, h = boss_h + 0.2);
    }
}

module rim() {
    // Four-sided registration rim around Jetson footprint
    // Only front + side walls; rear is open for cable exit
    x0 = jetson_origin_x - wall_t;
    y0 = jetson_origin_y - wall_t;
    w  = jetson_l + wall_t*2;
    d  = jetson_w + wall_t*2;
    difference() {
        translate([x0, y0, tray_t])
            cube([w, d, rim_h]);
        // Hollow out inside (Jetson pocket)
        translate([jetson_origin_x, jetson_origin_y, tray_t - 0.1])
            cube([jetson_l + 0.5, jetson_w + 0.5, rim_h + 0.2]);
        // Open rear wall for cable exit
        translate([x0 - 0.1, y0 + d - wall_t - 0.1, tray_t - 0.1])
            cube([w + 0.2, wall_t + 0.2, rim_h + 0.2]);
    }
}

// ── Main tray ────────────────────────────────────────────────
module jetson_tray() {
    difference() {
        union() {
            // Base plate
            cube([tray_l, tray_w, tray_t]);

            // Registration rim
            rim();

            // Jetson standoff bosses
            boss(jetson_hole(0,0)[0], jetson_hole(0,0)[1]);
            boss(jetson_hole(0,1)[0], jetson_hole(0,1)[1]);
            boss(jetson_hole(1,0)[0], jetson_hole(1,0)[1]);
            boss(jetson_hole(1,1)[0], jetson_hole(1,1)[1]);
        }

        // ── Rover mounting holes through base ───────────────
        for (h = rover_m3)
            translate([rover_origin_x + h[0],
                       rover_origin_y + h[1],
                       -0.1])
                cylinder(d = rover_hole_d, h = tray_t + 0.2);

        // ── Vent slots (cooling airflow for Jetson fan) ──────
        // 3 slots centred under Jetson footprint
        for (i = [0:2])
            translate([jetson_origin_x + 10 + i*28,
                       jetson_origin_y + 8,
                       -0.1])
                cube([16, jetson_w - 16, tray_t + 0.2]);

        // ── CSI ribbon cable slot (rear centre) ─────────────
        translate([csi_slot_x, tray_w - csi_slot_d, -0.1])
            cube([csi_slot_w, csi_slot_d + 0.2, tray_t + 0.2]);

        // ── GPIO wire channel (rear left) ───────────────────
        translate([gpio_slot_x, tray_w - 8, -0.1])
            cube([gpio_slot_w, 8.2, tray_t + 0.2]);

        // ── DC barrel jack clearance (right side notch) ─────
        // Barrel jack on +X edge of Jetson, ~centre of board Y
        translate([tray_l - 0.1,
                   tray_w/2 - barrel_w/2,
                   tray_t - barrel_h/2])
            cube([1.5, barrel_w, barrel_h]);

        // ── Corner chamfers (cosmetic + print quality) ───────
        chamf = 3;
        for (cx = [0, tray_l], cy = [0, tray_w])
            translate([cx, cy, -0.1])
            rotate([0, 0, cx==0 ? (cy==0?45:315) : (cy==0?135:225)])
            translate([-chamf, 0, 0])
                cube([chamf*2, chamf*2, tray_t+0.2]);
    }
}

jetson_tray();

// ── Debug: visualise Jetson footprint (comment out before exporting) ──
//color("blue", 0.15)
//translate([jetson_origin_x, jetson_origin_y, tray_t + 5])
//    cube([jetson_l, jetson_w, 1]);
