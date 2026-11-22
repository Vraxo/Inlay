namespace Inlay.Cli;

public sealed record CropRectangle(int X, int Y, int Width, int Height);

public sealed record RunOptions(
    string Input,
    string? Output,
    bool Recursive,
    bool Preview,
    CropRectangle? Rectangle,
    bool KeepCorners,
    bool Report,
    bool Overwrite);

public sealed record ParsedArguments(bool ShowHelp, string? Error, RunOptions? Options);

public static class Arguments
{
    public const string HelpText = """
        Inlay — extract a foreground image from a screenshot

        Usage:
          inlay <image>
          inlay <directory> [--recursive] [--out <directory>]

        Options:
          --out <path>          Output file (single input) or output directory (batch)
          --recursive            Include subdirectories
          --preview              Write a diagnostic JPEG preview
          --rect x,y,w,h         Use explicit crop coordinates in source pixels
          --corners auto|keep    Auto-mask rounded corners (default: auto), or keep pixels
          --report               Write a JSON report beside each output
          --overwrite            Replace existing output files
          -h, --help             Show help

        Supported inputs: PNG, JPG/JPEG, WebP
        Review and rejected detections are never exported as extracted images.
        """;

    public static ParsedArguments Parse(string[] args)
    {
        if (args.Length == 0 || args.Any(x => x is "-h" or "--help"))
            return new ParsedArguments(true, null, null);

        string? input = null;
        string? output = null;
        var recursive = false;
        var preview = false;
        var keepCorners = false;
        var report = false;
        var overwrite = false;
        CropRectangle? rectangle = null;

        for (var i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            if (!arg.StartsWith("-", StringComparison.Ordinal))
            {
                if (input is not null)
                    return new ParsedArguments(false, "Only one input path may be supplied.", null);
                input = arg;
                continue;
            }

            string Next()
            {
                if (++i >= args.Length)
                    throw new ArgumentException($"Option '{arg}' requires a value.");
                return args[i];
            }

            try
            {
                switch (arg)
                {
                    case "--out": output = Next(); break;
                    case "--recursive": recursive = true; break;
                    case "--preview": preview = true; break;
                    case "--report": report = true; break;
                    case "--overwrite": overwrite = true; break;
                    case "--rect":
                        var parts = Next().Split(',');
                        if (parts.Length != 4 || !parts.All(p => int.TryParse(p, out _)))
                            return new ParsedArguments(false, "--rect expects x,y,width,height as integers.", null);
                        var values = parts.Select(int.Parse).ToArray();
                        if (values[0] < 0 || values[1] < 0 || values[2] <= 0 || values[3] <= 0)
                            return new ParsedArguments(false, "--rect coordinates must be non-negative and dimensions positive.", null);
                        rectangle = new CropRectangle(values[0], values[1], values[2], values[3]);
                        break;
                    case "--corners":
                        var mode = Next();
                        if (mode is not ("auto" or "keep"))
                            return new ParsedArguments(false, "--corners must be 'auto' or 'keep'.", null);
                        keepCorners = mode == "keep";
                        break;
                    default: return new ParsedArguments(false, $"Unknown option '{arg}'.", null);
                }
            }
            catch (ArgumentException exception)
            {
                return new ParsedArguments(false, exception.Message, null);
            }
        }

        if (string.IsNullOrWhiteSpace(input))
            return new ParsedArguments(false, "An input file or directory is required.", null);

        return new ParsedArguments(false, null,
            new RunOptions(input, output, recursive, preview, rectangle, keepCorners, report, overwrite));
    }
}
