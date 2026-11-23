using System.Text.Json;
using Inlay.Detection;
using Inlay.Extraction;
using Inlay.Reporting;
using OpenCvSharp;

namespace Inlay.Processing;

public sealed record CropRectangle(int X, int Y, int Width, int Height);

public sealed record BatchOptions(
    string Input,
    string? Output,
    bool Recursive,
    bool Preview,
    CropRectangle? Rectangle,
    bool KeepCorners,
    bool Report,
    bool Overwrite);

public enum BatchItemStatus
{
    Extracted,
    Review,
    Rejected,
    Skipped,
    Failed
}

public sealed record BatchItemResult(
    string Input,
    BatchItemStatus Status,
    string? OutputPath = null,
    double? Confidence = null,
    string? Message = null);

public sealed record BatchSummary(
    int Scanned,
    int Extracted,
    int Review,
    int Rejected,
    int Failed,
    IReadOnlyList<BatchItemResult> Items);

public sealed class BatchProcessor
{
    private static readonly HashSet<string> SupportedExtensions = new(StringComparer.OrdinalIgnoreCase)
        { ".png", ".jpg", ".jpeg", ".webp" };

    private readonly ImageDetector detector = new();
    private readonly ImageExtractor extractor = new();

    private static bool IsWithinDirectory(string path, string directory)
    {
        var relative = Path.GetRelativePath(directory, Path.GetFullPath(path));
        return relative == "." || (relative != ".." &&
            !relative.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal) &&
            !Path.IsPathRooted(relative));
    }

    public async Task<BatchSummary> RunAsync(BatchOptions options)
    {
        var input = Path.GetFullPath(options.Input);
        if (!File.Exists(input) && !Directory.Exists(input))
            throw new FileNotFoundException($"Input path does not exist: {input}");

        var isDirectory = Directory.Exists(input);
        if (!isDirectory && !SupportedExtensions.Contains(Path.GetExtension(input)))
            throw new InvalidOperationException("Input must be a PNG, JPEG, or WebP image.");

        var outputRoot = options.Output is null
            ? isDirectory ? Path.Combine(input, "inlay") : Path.GetDirectoryName(input)!
            : Path.GetFullPath(options.Output);

        var outputIsNested = isDirectory &&
            !string.Equals(Path.GetFullPath(outputRoot), input, StringComparison.OrdinalIgnoreCase) &&
            IsWithinDirectory(outputRoot, input);

        var files = isDirectory
            ? (options.Recursive ? Directory.EnumerateFiles(input, "*", SearchOption.AllDirectories) : Directory.EnumerateFiles(input))
                .Where(path => SupportedExtensions.Contains(Path.GetExtension(path)))
                .Where(path => !outputIsNested || !IsWithinDirectory(path, outputRoot))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase).ToArray()
            : new[] { input };

        if (files.Length == 0)
            throw new InvalidOperationException("No supported image files were found.");

        var counts = new int[5];
        var items = new List<BatchItemResult>(files.Length);
        foreach (var file in files)
        {
            counts[0]++;
            try
            {
                var relative = isDirectory ? Path.GetRelativePath(input, file) : Path.GetFileName(file);
                var output = isDirectory
                    ? Path.Combine(outputRoot, Path.ChangeExtension(relative, ".inlay.png"))
                    : options.Output is null ? Path.Combine(outputRoot, Path.GetFileNameWithoutExtension(file) + ".inlay.png") : outputRoot;

                var previewPath = Path.ChangeExtension(output, ".preview.jpg");
                var reportPath = Path.ChangeExtension(output, ".json");

                if (!options.Overwrite &&
                    (File.Exists(output) || (options.Preview && File.Exists(previewPath)) || (options.Report && File.Exists(reportPath))))
                {
                    counts[4]++;
                    items.Add(new BatchItemResult(relative, BatchItemStatus.Skipped,
                        Message: "one or more output files already exist (use --overwrite)"));
                    continue;
                }

                using var source = Cv2.ImRead(file, ImreadModes.Color);
                if (source.Empty())
                    throw new InvalidDataException("OpenCV could not decode this image.");

                DetectionResult detection;
                if (options.Rectangle is not null)
                {
                    var rect = options.Rectangle;
                    if (rect.X + rect.Width > source.Width || rect.Y + rect.Height > source.Height)
                        throw new ArgumentOutOfRangeException(nameof(options.Rectangle), "Manual rectangle exceeds the source image dimensions.");
                    var bounds = new Rect(rect.X, rect.Y, rect.Width, rect.Height);
                    detection = new DetectionResult(DetectionStatus.Extracted,
                        new DetectionCandidate(bounds, 1, 1, 1, 1, 1, 1, 0, 0), null, "Manual crop coordinates supplied.");
                }
                else
                {
                    detection = detector.Detect(source);
                }

                if (options.Preview)
                    extractor.SavePreview(source, detection, previewPath);

                string? savedOutput = null;
                if (detection.Status == DetectionStatus.Extracted && detection.Best is not null)
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(output)!);
                    extractor.Save(source, detection.Best.Bounds, output, detection.Best.CornerRadius, options.KeepCorners);
                    savedOutput = output;
                    counts[1]++;
                    items.Add(new BatchItemResult(relative, BatchItemStatus.Extracted,
                        output, detection.Best.Confidence));
                }
                else if (detection.Status == DetectionStatus.Review)
                {
                    counts[2]++;
                    items.Add(new BatchItemResult(relative, BatchItemStatus.Review, Message: detection.Reason));
                }
                else
                {
                    counts[3]++;
                    items.Add(new BatchItemResult(relative, BatchItemStatus.Rejected, Message: detection.Reason));
                }

                if (options.Report)
                {
                    var best = detection.Best;
                    var report = new DetectionReport(
                        file,
                        detection.Status.ToString().ToLowerInvariant(),
                        best is null ? null : new BoundsReport(best.Bounds.X, best.Bounds.Y, best.Bounds.Width, best.Bounds.Height),
                        new CornersReport(best?.CornerRadius > 0, best?.CornerRadius ?? 0, savedOutput is not null && !options.KeepCorners && (best?.CornerRadius ?? 0) > 0),
                        best?.Confidence,
                        detection.Reason,
                        savedOutput);
                    Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
                    await File.WriteAllTextAsync(reportPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
                }
            }
            catch (Exception exception)
            {
                counts[4]++;
                items.Add(new BatchItemResult(Path.GetFileName(file), BatchItemStatus.Failed,
                    Message: exception.Message));
            }
        }

        return new BatchSummary(counts[0], counts[1], counts[2], counts[3], counts[4], items);
    }
}
