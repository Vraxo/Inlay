using System.Text.Json;
using Inlay.Detection;
using OpenCvSharp;

namespace Inlay.Cli;

internal static class BenchmarkRunner
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
        PropertyNameCaseInsensitive = true
    };

    internal static int Run(string manifestPath)
    {
        var fullManifestPath = Path.GetFullPath(manifestPath);
        var manifest = JsonSerializer.Deserialize<BenchmarkManifest>(
            File.ReadAllText(fullManifestPath), JsonOptions);

        if (manifest?.Cases is not { Count: > 0 })
            throw new InvalidDataException("The benchmark manifest must contain at least one case.");

        var minimumIou = manifest.MinimumIou ?? 0.90;
        if (minimumIou is <= 0 or > 1)
            throw new InvalidDataException("minimumIou must be greater than 0 and no greater than 1.");

        var manifestDirectory = Path.GetDirectoryName(fullManifestPath)!;
        var detector = new ImageDetector();
        var results = new List<BenchmarkCaseResult>(manifest.Cases.Count);

        foreach (var testCase in manifest.Cases)
            results.Add(EvaluateCase(testCase, manifestDirectory, detector, minimumIou));

        var positiveCases = results.Where(result => result.Expected is not null).ToArray();
        var detectedCases = positiveCases.Where(result => result.Actual is not null).ToArray();
        var geometryPasses = positiveCases.Count(result => result.Iou >= minimumIou);
        var automaticPasses = positiveCases.Count(result => result.AutomaticallyExtracted);
        var falsePositives = results.Count(result => result.Expected is null && result.Actual is not null);
        var passedCases = results.Count(result => result.Passed);

        var report = new BenchmarkReport(
            DateTimeOffset.UtcNow,
            minimumIou,
            results.Count,
            positiveCases.Length,
            detectedCases.Length,
            geometryPasses,
            automaticPasses,
            falsePositives,
            results);

        var reportPath = Path.Combine(manifestDirectory, "inlay-benchmark-results.json");
        File.WriteAllText(reportPath, JsonSerializer.Serialize(report, JsonOptions));

        Console.WriteLine($"Cases: {results.Count}  |  Passed: {passedCases}  |  Failed: {results.Count - passedCases}");
        Console.WriteLine($"Foregrounds: {positiveCases.Length}  |  Found: {detectedCases.Length}  |  IoU ≥ {minimumIou:P0}: {geometryPasses}  |  Auto-extracted: {automaticPasses}");
        Console.WriteLine($"Negative cases: {results.Count - positiveCases.Length}  |  False positives: {falsePositives}");
        Console.WriteLine($"Report: {reportPath}");

        foreach (var result in results.Where(result => !result.Passed))
            Console.WriteLine($"  FAIL  {result.Image}: {result.Message}");

        return results.All(result => result.Passed) ? 0 : 1;
    }

    internal static double IntersectionOverUnion(BenchmarkBounds expected, Rect actual)
    {
        var left = Math.Max(expected.X, actual.X);
        var top = Math.Max(expected.Y, actual.Y);
        var right = Math.Min(expected.X + expected.Width, actual.Right);
        var bottom = Math.Min(expected.Y + expected.Height, actual.Bottom);
        var intersection = (double)Math.Max(0, right - left) * Math.Max(0, bottom - top);
        var expectedArea = (double)expected.Width * expected.Height;
        var actualArea = (double)actual.Width * actual.Height;
        var union = expectedArea + actualArea - intersection;
        return union <= 0 ? 0 : intersection / union;
    }

    private static BenchmarkCaseResult EvaluateCase(
        BenchmarkCase testCase,
        string manifestDirectory,
        ImageDetector detector,
        double minimumIou)
    {
        var imagePath = Path.IsPathRooted(testCase.Image)
            ? testCase.Image
            : Path.GetFullPath(Path.Combine(manifestDirectory, testCase.Image));
        var imageName = testCase.Image;

        if (!File.Exists(imagePath))
            return BenchmarkCaseResult.Failed(imageName, testCase.Expected, $"Image not found: {imageName}");

        using var image = Cv2.ImRead(imagePath, ImreadModes.Color);
        if (image.Empty())
            return BenchmarkCaseResult.Failed(imageName, testCase.Expected, "Image could not be decoded.");

        if (testCase.Expected is { } expected &&
            (expected.X < 0 || expected.Y < 0 || expected.Width <= 0 || expected.Height <= 0 ||
             (long)expected.X + expected.Width > image.Width ||
             (long)expected.Y + expected.Height > image.Height))
            return BenchmarkCaseResult.Failed(imageName, testCase.Expected, "Expected rectangle is outside the source image.");

        var detection = detector.Detect(image);
        var actual = detection.Best?.Bounds;
        if (testCase.Expected is null)
            return new BenchmarkCaseResult(
                imageName, null, actual is { } found ? new BenchmarkBounds(found.X, found.Y, found.Width, found.Height) : null,
                actual is null ? null : 0, null, detection.Status.ToString(), actual is null,
                actual is null ? "Correctly found no foreground." : "False positive: a foreground candidate was returned.");

        if (actual is null)
            return new BenchmarkCaseResult(
                imageName, testCase.Expected, null, 0, null, detection.Status.ToString(),
                false, "No foreground candidate was found.");

        var iou = IntersectionOverUnion(testCase.Expected, actual.Value);
        var edgeError = MeanEdgeError(testCase.Expected, actual.Value);
        var geometryPassed = iou >= minimumIou;
        var automatic = detection.Status == DetectionStatus.Extracted && geometryPassed;
        var passed = automatic;

        var message = !geometryPassed
            ? $"IoU {iou:P1} below {minimumIou:P0}; mean edge error {edgeError:0.0}px."
            : detection.Status != DetectionStatus.Extracted
                ? $"Geometry passed (IoU {iou:P1}), but detector status is {detection.Status}."
                : $"Automatic extraction passed (IoU {iou:P1}).";

        return new BenchmarkCaseResult(
            imageName,
            testCase.Expected,
            new BenchmarkBounds(actual.Value.X, actual.Value.Y, actual.Value.Width, actual.Value.Height),
            iou,
            edgeError,
            detection.Status.ToString(),
            passed,
            message);
    }

    private static double MeanEdgeError(BenchmarkBounds expected, Rect actual)
    {
        var expectedRight = expected.X + expected.Width;
        var expectedBottom = expected.Y + expected.Height;
        return (Math.Abs(expected.X - actual.X) +
                Math.Abs(expected.Y - actual.Y) +
                Math.Abs(expectedRight - actual.Right) +
                Math.Abs(expectedBottom - actual.Bottom)) / 4.0;
    }

    private sealed record BenchmarkManifest
    {
        public double? MinimumIou { get; init; }
        public List<BenchmarkCase> Cases { get; init; } = [];
    }

    private sealed record BenchmarkCase
    {
        public string Image { get; init; } = "";
        public BenchmarkBounds? Expected { get; init; }
    }

    private sealed record BenchmarkBounds(int X, int Y, int Width, int Height);

    private sealed record BenchmarkCaseResult(
        string Image,
        BenchmarkBounds? Expected,
        BenchmarkBounds? Actual,
        double? Iou,
        double? MeanEdgeErrorPixels,
        string Status,
        bool Passed,
        string Message)
    {
        internal static BenchmarkCaseResult Failed(string image, BenchmarkBounds? expected, string message) =>
            new(image, expected, null, expected is null ? null : 0, null, "Failed", false, message);
    }

    private sealed record BenchmarkReport(
        DateTimeOffset GeneratedAt,
        double MinimumIou,
        int Cases,
        int Foregrounds,
        int Found,
        int GeometryPasses,
        int AutomaticallyExtracted,
        int FalsePositives,
        IReadOnlyList<BenchmarkCaseResult> Results);
}
