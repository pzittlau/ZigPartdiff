const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;

const Method = enum {
    gauss_seidel,
    jacobi,
};

const InferenceFunction = enum {
    none,
    sin,
};

const TerminationCondition = union(enum) {
    precision: f64,
    iteration: u64,
};

const CalculationArguments = struct {
    N: u64,
    num_matrices: u64,
    h: f64,
    M: []f64,
};

const CalculationResults = struct {
    m: u64,
    stat_iteration: u64,
    stat_precision: f64,
};

const Options = struct {
    num_threads: u64,
    method: Method,
    interlines: u64,
    inference_function: InferenceFunction,
    termination: TerminationCondition,
};

const ArgumentError = error{
    insufficient_arguments,
    help,
    not_enough_threads,
    to_many_threads,
    unknown_method,
    to_many_interlines,
    unknown_interference_function,
    unknown_termination_condition,
    not_enough_iterations,
    to_many_iterations,
};

const max_interlines = 100000;
const max_iterations = 200000;
const max_threads = 1024;

var timer: std.time.Timer = undefined;
var time_ns: u64 = undefined;

fn usage(name: []const u8) void {
    print("Usage: {s} [num] [method] [lines] [func] [term] [prec/iter]\n\n", .{name});
    print("  - num:       number of threads (1 .. {})\n", .{max_threads});
    print("  - method:    calculation method (1 .. 2)\n", .{});
    print("                 {}: Gauß-Seidel\n", .{@intFromEnum(Method.gauss_seidel)});
    print("                 {}: Jacobi\n", .{@intFromEnum(Method.jacobi)});
    print("  - lines:     number of interlines (0 .. {})\n", .{max_interlines});
    print("                 matrix size = (interlines * 8) + 8\n", .{});
    print("  - func:      interference function (1 .. 2)\n", .{});
    print("                 {}: f(x,y) = 0\n", .{@intFromEnum(InferenceFunction.none)});
    print("                 {}: f(x,y) = 2 * pi^2 * sin(pi * x) * sin(pi * y)\n", .{@intFromEnum(InferenceFunction.sin)});
    print("  - term:      termination condition (1 .. 2)\n", .{});
    print("                 {}: sufficient precision\n", .{@intFromEnum(TerminationCondition.precision)});
    print("                 {}: number of iterations\n", .{@intFromEnum(TerminationCondition.iteration)});
    print("  - prec/iter: depending on term:\n", .{});
    print("                 precision:  1e-4 .. 1e-20\n", .{});
    print("                 iterations:    1 .. {}\n\n", .{max_iterations});
    print("Example: {s} 1 2 100 1 2 100\n", .{name});
}

fn askParams(allocator: Allocator) !Options {
    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);

    if (argv.len < 7) {
        usage(argv[0]);
        return ArgumentError.insufficient_arguments;
    }

    if (std.mem.eql(u8, argv[1], "-h") or std.mem.eql(u8, argv[1], "-?")) {
        usage(argv[0]);
        return ArgumentError.help;
    }

    const parseInt = std.fmt.parseInt;
    const threads = switch (try parseInt(u64, argv[1], 10)) {
        0 => return ArgumentError.not_enough_threads,
        1...max_threads => |v| v,
        else => return ArgumentError.to_many_threads,
    };
    const method: Method = @enumFromInt(try parseInt(u64, argv[2], 10) - 1);
    const interlines = switch (try parseInt(u64, argv[3], 10)) {
        0...max_interlines => |v| v,
        else => return ArgumentError.to_many_interlines,
    };
    const inf_func: InferenceFunction = @enumFromInt(try parseInt(u64, argv[4], 10) - 1);
    const termination: TerminationCondition = blk: {
        if (try parseInt(u64, argv[5], 10) - 1 == @intFromEnum(TerminationCondition.precision)) {
            // TODO: figure out how to parse a float in scientific notation
            break :blk .{ .precision = try std.fmt.parseFloat(f64, argv[6]) };
        } else if (try parseInt(u64, argv[5], 10) - 1 == @intFromEnum(TerminationCondition.iteration)) {
            switch (try parseInt(u64, argv[6], 10)) {
                0 => return ArgumentError.not_enough_iterations,
                1...max_iterations => |v| break :blk .{ .iteration = v },
                else => return ArgumentError.to_many_iterations,
            }
        } else {
            return ArgumentError.unknown_termination_condition;
        }
    };

    return Options{
        .num_threads = threads,
        .method = method,
        .interlines = interlines,
        .inference_function = inf_func,
        .termination = termination,
    };
}

fn initVariables(arguments: *CalculationArguments, options: Options) void {
    arguments.N = (options.interlines * 8) + 8;
    arguments.num_matrices = switch (options.method) {
        .gauss_seidel => 1,
        .jacobi => 2,
    };
    arguments.h = 1.0 / @as(f64, @floatFromInt(arguments.N));
}

fn getIndex(matrix: u64, row: u64, column: u64, N: u64) u64 {
    return column + (N + 1) * (row + (N + 1) * matrix);
}

fn initMatrices(arguments: *CalculationArguments, options: Options) void {
    @memset(arguments.M, 0);

    if (options.inference_function == .sin) {
        return;
    }

    const h = arguments.h;
    const N = arguments.N;
    const M = arguments.M;
    for (0..arguments.num_matrices) |g| {
        for (0..N + 1) |i| {
            M[getIndex(g, i, 0, N)] = 1.0 - (h * @as(f64, @floatFromInt(i)));
            M[getIndex(g, i, N, N)] = h * @as(f64, @floatFromInt(i));
            M[getIndex(g, 0, i, N)] = 1.0 - (h * @as(f64, @floatFromInt(i)));
            M[getIndex(g, N, i, N)] = h * @as(f64, @floatFromInt(i));
        }
    }
}

fn calculate(arguments: *CalculationArguments, options: Options) CalculationResults {
    var star: f64 = undefined;
    var residuum: f64 = undefined;
    var max_residuum: f64 = undefined;
    var m1: u1 = 0;
    var m2: u1 = if (options.method == .jacobi) 1 else 0;

    const pih: f64 = if (options.inference_function == .sin) std.math.pi * arguments.h else 0;
    const fpisin: f64 = if (options.inference_function == .sin) 0.25 * 2 * std.math.pi * arguments.h * arguments.h else 0;
    const M = arguments.M;
    const N = arguments.N;

    var current_iteration: u64 = 1;
    while (true) : (current_iteration += 1) {
        max_residuum = 0;
        for (1..N) |i| {
            const fpisin_i = if (options.inference_function == .sin) fpisin * std.math.sin(pih * @as(f64, @floatFromInt(i))) else 0;
            for (1..N) |j| {
                star = 0.25 * (M[getIndex(m2, i - 1, j, N)] +
                    M[getIndex(m2, i, j - 1, N)] +
                    M[getIndex(m2, i, j + 1, N)] +
                    M[getIndex(m2, i + 1, j, N)]);

                if (options.inference_function == .sin) {
                    star += fpisin_i * std.math.sin(pih * @as(f64, @floatFromInt(i)));
                }

                if (options.termination == .precision or current_iteration == options.termination.iteration) {
                    residuum = @abs(M[getIndex(m2, i, j, N)] - star);
                    max_residuum = if (residuum < max_residuum) max_residuum else residuum;
                }

                M[getIndex(m1, i, j, N)] = star;
            }
        }

        const temp = m1;
        m1 = m2;
        m2 = temp;

        switch (options.termination) {
            .iteration => |iteration| if (current_iteration >= iteration) {
                break;
            },
            .precision => |precision| if (max_residuum <= precision) {
                break;
            },
        }
    }

    return CalculationResults{
        .m = m2,
        .stat_iteration = current_iteration,
        .stat_precision = max_residuum,
    };
}

fn displayStatistics(arguments: CalculationArguments, results: CalculationResults, options: Options) void {
    print("Berechnungszeit:    {d:.6} s\n", .{@as(f64, @floatFromInt(time_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s))});
    print("Speicherbedarf:     {d:.6} MiB\n", .{@as(f64, @floatFromInt((arguments.N + 1) * (arguments.N + 1) * @sizeOf(f64) * arguments.num_matrices)) / 1024.0 / 1024.0});
    print("Berechnungsmethode: ", .{});
    switch (options.method) {
        .gauss_seidel => print("Gauß-Seidel\n", .{}),
        .jacobi => print("Jacobi\n", .{}),
    }
    print("Interlines:         {}\n", .{options.interlines});
    print("Störfunktion:       ", .{});
    switch (options.inference_function) {
        .none => print("f(x,y) = 0\n", .{}),
        .sin => print("f(x,y) = 2 * pi^2 * sin(pi * x) * sin(pi * y)\n", .{}),
    }
    print("Terminierung:       ", .{});
    switch (options.termination) {
        .iteration => print("Anzahl der Iterationen\n", .{}),
        .precision => print("Hinreichende Genauigkeit\n", .{}),
    }
    print("Anzahl Iterationen: {}\n", .{results.stat_iteration});
    print("Norm des Fehlers:   {}\n\n", .{results.stat_precision});
}

fn displayMatrix(arguments: CalculationArguments, results: CalculationResults, options: Options) void {
    print("Matrix:\n", .{});

    for (0..9) |y| {
        for (0..9) |x| {
            print("{d:7.4}", .{arguments.M[getIndex(results.m, y * (options.interlines + 1), x * (options.interlines + 1), arguments.N)]});
        }
        print("\n", .{});
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const status = gpa.deinit();
        if (status == .leak) @panic("MEMORY LEAK");
    }

    const options = try askParams(allocator);
    var arguments: CalculationArguments = undefined;

    initVariables(&arguments, options);

    const numElements = arguments.num_matrices * (arguments.N + 1) * (arguments.N + 1);
    arguments.M = try allocator.alloc(f64, numElements);
    defer allocator.free(arguments.M);
    initMatrices(&arguments, options);

    timer = try std.time.Timer.start();
    const results = calculate(&arguments, options);
    time_ns = timer.read();

    displayStatistics(arguments, results, options);
    displayMatrix(arguments, results, options);
}
