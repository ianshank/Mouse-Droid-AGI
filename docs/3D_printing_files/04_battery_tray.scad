// ============================================================
//  MSE-6 — Battery Tray (3× 18650 cells in series = 3S pack)
//  Mounts on Wave Rover expansion plate alongside Jetson tray.
//
//  18650 cell: Ø18.5mm × 65mm (with protection circuit ~70mm)
//  3S pack arranged side-by-side (not in-line) to minimise length
//  Pack footprint: 3 × 18.5mm + walls ≈ 62mm wide × 72mm long
//
//  Features:
//    – 3 cell tubes with spring-tab retention
//    – 2× Velcro strap slots (5mm wide) for belt security
//    – Vent slots between cells
//    – 2× M3 holes matching rover expansion plate holes
//    – Wire exit notch (rear) for battery leads / BMS board
//    – Low-profile: 22mm tall (cell OD + base = 18.5+3.5)
//
//  Wave Rover expansion plate: ~117 x 127mm
//  Jetson tray: 114 x 88mm  →  battery tray fills remaining space
// ============================================================
$fn = 48;

// Cell dimensions
cell_d   = 18.5;   // 18650 diameter (protected)
cell_l   = 70.0;   // 18650 length (protected)
n_cells  = 3;

// Wall / structure
wall     =  2.5;
base_t   =  3.0;
end_t    =  3.0;   // end wall thickness

// Tray outer dimensions
tray_w   = n_cells * cell_d + (n_cells + 1) * wall;  // ~62.5mm
tray_l   = cell_l + end_t * 2;                        // ~76mm
tray_h   = cell_d / 2 + base_t + 4;                  // ~18mm (half-tube cradle)

// Strap slots
strap_w  =  6.0;
strap_h  =  4.0;

// M3 mount holes (2, matching rover plate)
m3_d     =  3.4;

// Wire exit notch
wire_w   = 16.0;
wire_h   =  8.0;

module cell_tube(x, y) {
    // Half-tube cradle for one cell
    translate([x, end_t, base_t])
    difference() {
        // Solid half-cylinder saddle
        rotate([90, 0, 0]) {
            // Outer shell
            cylinder(d = cell_d + wall*2, h = -(cell_l), center=false);
        }
        // Cell bore (full cylinder - leaves half saddle after base diff)
        translate([0, -(cell_l+0.1), 0])
            rotate([90, 0, 0])
                cylinder(d = cell_d + 0.4, h = cell_l + 0.2);
        // Open top (cut upper half)
        translate([-(cell_d/2+wall+0.1), -(cell_l+0.1), 0])
            cube([cell_d + wall*2 + 0.2, cell_l + 0.2, cell_d]);
    }
}

module battery_tray() {
    difference() {
        union() {
            // Base plate
            cube([tray_w, tray_l, base_t]);

            // End walls
            cube([tray_w, end_t, tray_h]);
            translate([0, tray_l - end_t, 0])
                cube([tray_w, end_t, tray_h]);

            // Side walls
            cube([wall, tray_l, tray_h]);
            translate([tray_w - wall, 0, 0])
                cube([wall, tray_l, tray_h]);

            // Cell dividers between cells
            for (i = [1 : n_cells-1])
                translate([i * (cell_d + wall), 0, 0])
                    cube([wall, tray_l, tray_h]);
        }

        // Cell bores (half-round cradles)
        for (i = [0 : n_cells-1]) {
            cx = wall + cell_d/2 + i * (cell_d + wall);
            // Cell bore
            translate([cx, end_t - 0.1, base_t + cell_d/2])
                rotate([-90, 0, 0])
                    cylinder(d = cell_d + 0.5, h = cell_l + 0.2);
        }

        // Vent slots between cells (base)
        for (i = [0 : n_cells-1]) {
            cx = wall + i * (cell_d + wall) + wall/2;
            translate([cx + wall/2, tray_l * 0.3, -0.1])
                cube([cell_d - wall, tray_l * 0.4, base_t + 0.2]);
        }

        // Strap slots (2 cross slots through side walls)
        for (sy = [tray_l*0.28, tray_l*0.72])
            translate([-0.1, sy - strap_w/2, tray_h - strap_h])
                cube([tray_w + 0.2, strap_w, strap_h + 0.1]);

        // Wire exit notch (rear wall)
        translate([tray_w/2 - wire_w/2, tray_l - end_t - 0.1, base_t])
            cube([wire_w, end_t + 0.2, wire_h]);

        // M3 mount holes (through base, near rover plate holes)
        translate([tray_w * 0.2, tray_l * 0.5, -0.1])
            cylinder(d = m3_d, h = base_t + 0.2);
        translate([tray_w * 0.8, tray_l * 0.5, -0.1])
            cylinder(d = m3_d, h = base_t + 0.2);
    }
}

battery_tray();
