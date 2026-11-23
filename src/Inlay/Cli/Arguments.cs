using System.CommandLine;
using Inlay.Processing;

namespace Inlay.Cli;

public sealed record CropRectangle(int X, int Y, int Width, int Height);

public static class Arguments
{
    public static RootCommand CreateCommand()
    {
        var inputArgument = new Argument<string?>("input")
        {
            Arity = ArgumentArity.ZeroOrOne,
            Description = "An image file or directory to process"
        };
        var outputOption = new Option<string?>("--out")
        {
            Description = "Output file (single input) or output directory (batch)"
        };
        var recursiveOption = new Option<bool>("--recursive")
        {
            Description = "Include subdirectories"
        };
        var previewOption = new Option<bool>("--preview")
        {
            Description = "Write a diagnostic JPEG preview"
        };
        var rectangleOption = new Option<string?>("--rect")
        {
            Description = "Use explicit crop coordinates in source pixels: x,y,width,height"
        };
        var cornersOption = new Option<string>("--corners")
        {
            DefaultValueFactory = _ => "auto",
            Description = "Rounded corners: auto (default) or keep"
        };
        var reportOption = new Option<bool>("--report")
        {
            Description = "Write a JSON report beside each output"
        };
        var overwriteOption = new Option<bool>("--overwrite")
        {
            Description = "Replace existing output files"
        };

        var command = new RootCommand("Extract a foreground image from a screenshot");
        command.Arguments.Add(inputArgument);
        command.Options.Add(outputOption);
        command.Options.Add(recursiveOption);
        command.Options.Add(previewOption);
        command.Options.Add(rectangleOption);
        command.Options.Add(cornersOption);
        command.Options.Add(reportOption);
        command.Options.Add(overwriteOption);

        command.SetAction(async parseResult =>
        {
            var input = parseResult.GetValue(inputArgument);
            if (string.IsNullOrWhiteSpace(input))
                return 2;

            var corners = parseResult.GetValue(cornersOption);
            if (corners is not ("auto" or "keep"))
            {
                Console.Error.WriteLine("inlay: --corners must be 'auto' or 'keep'.");
                return 2;
            }

            CropRectangle? rectangle = null;
            var rectangleValue = parseResult.GetValue(rectangleOption);
            if (rectangleValue is not null && !TryParseRectangle(rectangleValue, out rectangle))
            {
                Console.Error.WriteLine("inlay: --rect expects non-negative x,y coordinates and positive width,height integers.");
                return 2;
            }

            try
            {
                var summary = await new BatchProcessor().RunAsync(new RunOptions(
                    input,
                    parseResult.GetValue(outputOption),
                    parseResult.GetValue(recursiveOption),
                    parseResult.GetValue(previewOption),
                    rectangle,
                    corners == "keep",
                    parseResult.GetValue(reportOption),
                    parseResult.GetValue(overwriteOption)));

                Console.WriteLine();
                Console.WriteLine($"Scanned {summary.Scanned}  |  Extracted {summary.Extracted}  |  Review {summary.Review}  |  Rejected {summary.Rejected}  |  Failed {summary.Failed}");
                return summary.Failed > 0 || summary.Rejected > 0 || summary.Review > 0 ? 1 : 0;
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine($"inlay: {exception.Message}");
                return 2;
            }
        });

        return command;
    }

    public static bool TryParseRectangle(string value, out CropRectangle? rectangle)
    {
        rectangle = null;
        var parts = value.Split(',');
        if (parts.Length != 4 ||
            !int.TryParse(parts[0], out var x) ||
            !int.TryParse(parts[1], out var y) ||
            !int.TryParse(parts[2], out var width) ||
            !int.TryParse(parts[3], out var height) ||
            x < 0 || y < 0 || width <= 0 || height <= 0)
            return false;

        rectangle = new CropRectangle(x, y, width, height);
        return true;
    }
}
