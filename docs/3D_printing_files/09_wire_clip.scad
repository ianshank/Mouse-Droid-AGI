// ============================================================
//  MSE-6 — Wire Routing Clip  (print 6-8×)
//  Snap-over cable management clips that screw to the
//  Wave Rover expansion plate or shell interior.
//
//  Two variants controlled by WIRE_D:
//    small (4.0mm) : HC-SR04 4-wire bundle / single JST wire
//    large (8.0mm) : USB-serial cable to ESP32 / power wires
//
//  M3 screw mount, base 12×10mm
// ============================================================
$fn = 32;

WIRE_D = 8.0;   // change to 4.0 for small clip

// Clip dimensions
wire_d   = WIRE_D;
wall     = 1.8;
base_w   = 14.0;
base_d   = 10.0;
base_t   =  3.0;
arch_gap = wire_d * 0.45;   // snap opening (slightly less than radius)
m3_d     =  3.4;
m3_csink =  6.0;

// Arch dimensions
arch_od  = wire_d + wall*2;
arch_h   = arch_od / 2 + wall;

module wire_clip() {
    difference() {
        union() {
            // Base plate
            cube([base_w, base_d, base_t]);
            // Arch centred on base
            translate([base_w/2, base_d/2, base_t])
            difference() {
                // Full arch cylinder
                cylinder(d = arch_od, h = arch_h);
                // Inner bore
                translate([0, 0, -0.1])
                    cylinder(d = wire_d, h = arch_h + 0.2);
                // Snap gap (bottom opening for wire)
                translate([-arch_gap/2, -arch_od/2 - 0.1, -0.1])
                    cube([arch_gap, arch_od/2 + 0.2, arch_h + 0.2]);
                // Snap chamfer (makes wire snap in)
                translate([-arch_gap/2 - 1, -arch_od/2 - 0.1, arch_h - 2])
                    rotate([0, 45, 0])
                        cube([1.5, arch_od/2 + 0.2, 1.5]);
                translate([arch_gap/2 - 0.5, -arch_od/2 - 0.1, arch_h - 2])
                    rotate([0, 45, 0])
                        cube([1.5, arch_od/2 + 0.2, 1.5]);
            }
        }
        // M3 screw hole (centre of base)
        translate([base_w/2, base_d/2, -0.1]) {
            cylinder(d = m3_d, h = base_t + 0.2);
            cylinder(d = m3_csink, h = 1.8);
        }
    }
}

wire_clip();
