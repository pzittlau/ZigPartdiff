# ZigPartdiff

This is a Zig port of a C Program to calculate grid based differential 
equations and was made to learn a bit of Zig.

# Usage

Build the code with `zig build` and optionally optimization flags as per `zig
build -h` for instance `zig build -Doptimize=ReleaseFast`.
Then run the binary `zig-out/bin/zigpartdiff`.

Example: `zig build -Doptimize=ReleaseFast && ./zig-out/bin/zigpartdiff 1 2 100
1 2 100`

To understand what the arguments mean run with `-h` or `-?`.

Alternatively compile and run with `zig build run -- 1 2 100 1 2 100`.

# License

Based on the original code it's licensed under GPLv3.
