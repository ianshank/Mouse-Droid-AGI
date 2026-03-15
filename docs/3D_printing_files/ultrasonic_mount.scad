// ============================================================
//  MSE-6 Mouse Droid — HC-SR04 Ultrasonic Sensor Mount
//  Fits mse6_v6_bottom.stl (210 x 184 x 64 mm chassis)
//
//  Sensor specs (HC-SR04):
//    PCB     : 45 x 20 mm
//    Cans    : 2 × Ø16 mm transducers, 26 mm centre spacing
//    Height  : ~15 mm total (PCB + can protrusion)
//    Pins    : 4-pin header on one short edge
//    M2 holes: 40 mm spacing along PCB long axis, 1 mm from edge
//
//  Design:
//    - Single bracket mounts one HC-SR04, front-facing
//    - Can be mirrored / stacked for additional sensors
//    - Two M3 ears mate to chassis boss holes:
//        Left boss  ≈ (27.3, 54.3,  42 mm)
//        Right boss ≈ (27.3, 126.3, 42 mm)
//      → 72 mm Y span; place mount between these bosses
//    - Wire exit slot at rear
//    - Sensor is slightly angled down 5° to scan near-floor
//
//  Print: PLA, 0.2 mm layers, 3 walls
// ============================================================

$fn = 64;

// ── Parameters ───────────────────────────────────────────────

// HC-SR04 PCB
pcb_l         = 45.0;   // PCB length
pcb_w         = 20.0;   // PCB width
pcb_t         = 1.6;    // PCB thickness
can_dia       = 16.0;   // transducer can outer diameter
can_spacing   = 26.0;   // centre-to-centre of the two cans
can_cx        = pcb_l / 2;  // centre of first can from PCB left (approx)
// cans at  pcb_l/2 ± can_spacing/2 along X, at pcb_w/2 along Y
can_protrude  = 12.0;   // how far cans stick above PCB top face
pcb_m2_x      = 2.5;    // M2 hole dist from long edge
pcb_m2_span   = 40.0;   // M2 hole span along long axis
pcb_m2_d      = 2.2;    // M2 clearance diameter

// Cradle geometry
wall          = 2.0;
floor_t       = 2.5;
clip_h        = 3.5;    // retaining lip over PCB top face
pcb_clearance = 0.3;    // loose fit in X/Y

// Tilt
tilt_down     = 5;      // degrees nose-down

// Base plate
base_l        = 54.0;   // slightly wider than PCB
base_w        = 14.0;   // narrow — minimise chassis footprint
base_t        = 3.0;

// M3 mounting holes
m3_d          = 3.4;
m3_ear_r      = 5.0;
// Two holes along Y; you choose offset on chassis
m3_hole_y1    = 3.0;
m3_hole_y2    = base_w - 3.0;

// Cable / header slot
hdr_slot_w    = 12.0;   // header is ~8 mm wide, give clearance
hdr_slot_h    = 6.0;
hdr_pos_x     = 0;      // header is on the left short end of PCB

// ── Helper modules ───────────────────────────────────────────

module m3_ear_2d(x, y) {
    translate([x, y, 0])
    difference() {
        hull() {
            cylinder(r = m3_ear_r, h = base_t);
            translate([-m3_ear_r, 0, 0]) cube([m3_ear_r*2, 0.01, base_t]);
        }
        cylinder(d = m3_d, h = base_t + 1);
    }
}

// ── Cradle for one HC-SR04 ───────────────────────────────────
module hcsr04_cradle() {
    difference() {
        // Outer solid
        translate([-wall, -wall, 0])
            cube([pcb_l + wall*2 + pcb_clearance,
                  pcb_w + wall*2 + pcb_clearance,
                  floor_t + pcb_t + clip_h]);

        // PCB pocket (slightly oversize)
        translate([0, 0, floor_t])
            cube([pcb_l + pcb_clearance,
                  pcb_w + pcb_clearance,
                  pcb_t + clip_h + 0.1]);

        // Transducer cutouts — two circles through everything
        for (sign = [-1, 1])
            translate([pcb_l/2 + sign * can_spacing/2,
                       pcb_w/2,
                       -0.1])
                cylinder(d = can_dia + 0.8, h = floor_t + pcb_t + clip_h + 1);

        // Pin header slot on left short wall
        translate([-wall - 0.1,
                   pcb_w/2 - hdr_slot_w/2,
                   floor_t + pcb_t - 0.5])
            cube([wall + 1, hdr_slot_w, hdr_slot_h]);

        // Wire channel on back long wall
        translate([pcb_l/2 - hdr_slot_w/2,
                   pcb_w + wall*2 + pcb_clearance - wall - 0.1,
                   floor_t + pcb_t])
            cube([hdr_slot_w, wall + 1, hdr_slot_h]);

        // M2 screw access holes through cradle floor
        for (dx = [pcb_m2_x, pcb_l - pcb_m2_x])
            translate([dx, pcb_w/2, -0.1])
                cylinder(d = pcb_m2_d, h = floor_t + 1);
    }
}

// ── Full mount ───────────────────────────────────────────────
module ultrasonic_mount() {
    union() {
        // Base plate
        difference() {
            cube([base_l, base_w, base_t]);
            // Two inline M3 holes centred on base width
            translate([base_l * 0.22, base_w/2, -0.1]) cylinder(d = m3_d, h = base_t + 1);
            translate([base_l * 0.78, base_w/2, -0.1]) cylinder(d = m3_d, h = base_t + 1);
        }

        // M3 ears at front & rear for optional second mount row
        m3_ear_2d(base_l/2 - m3_ear_r*0.5, -m3_ear_r*1.5);
        translate([0, base_w + m3_ear_r*0.5, 0])
            m3_ear_2d(base_l/2 - m3_ear_r*0.5, 0);

        // Tilted cradle riser
        translate([(base_l - pcb_l)/2 - wall, 0, base_t])
        rotate([-tilt_down, 0, 0])
            hcsr04_cradle();
    }
}

ultrasonic_mount();
