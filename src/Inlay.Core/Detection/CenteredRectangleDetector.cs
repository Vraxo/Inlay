using OpenCvSharp;

namespace Inlay.Detection;

internal static class CenteredRectangleDetector
{
    private const int MaximumSearchDimension = 160;
    private const int MaximumCandidates = 60;
    private const int MaximumCandidatesPerAreaBand = 5;

    internal static IReadOnlyList<Rect> FindCandidates(Mat focusMap, Mat edges)
    {
        var scale = Math.Min(1.0, (double)MaximumSearchDimension / Math.Max(focusMap.Width, focusMap.Height));
        using var focus = new Mat();
        using var edgeMap = new Mat();

        if (scale < 1.0)
        {
            var size = new Size(
                Math.Max(1, (int)Math.Round(focusMap.Width * scale)),
                Math.Max(1, (int)Math.Round(focusMap.Height * scale)));
            Cv2.Resize(focusMap, focus, size, interpolation: InterpolationFlags.Area);
            Cv2.Resize(edges, edgeMap, size, interpolation: InterpolationFlags.Area);
        }
        else
        {
            focusMap.CopyTo(focus);
            edges.CopyTo(edgeMap);
        }

        if (Cv2.CountNonZero(edgeMap) == 0)
            return Array.Empty<Rect>();

        var candidates = new List<(Rect Bounds, double Score, double AreaRatio, double WidthRatio, double HeightRatio)>();
        var widthRatios = Enumerable.Range(6, 14).Select(value => value * 0.05);
        var heightRatios = Enumerable.Range(5, 15).Select(value => value * 0.05);
        var offsets = new[] { -0.10, -0.05, 0.0, 0.05, 0.10 };

        foreach (var widthRatio in widthRatios)
        {
            var width = (int)Math.Round(focus.Width * widthRatio);
            foreach (var heightRatio in heightRatios)
            {
                var height = (int)Math.Round(focus.Height * heightRatio);
                var areaRatio = widthRatio * heightRatio;
                var aspectRatio = (double)width / Math.Max(1, height);
                if (areaRatio is < 0.05 or > 0.90 || aspectRatio is < 0.30 or > 3.2)
                    continue;

                foreach (var offsetY in offsets)
                {
                    foreach (var offsetX in offsets)
                    {
                        var centerX = focus.Width / 2.0 + offsetX * focus.Width;
                        var centerY = focus.Height / 2.0 + offsetY * focus.Height;
                        var bounds = new Rect(
                            (int)Math.Round(centerX - width / 2.0),
                            (int)Math.Round(centerY - height / 2.0),
                            width,
                            height);

                        if (bounds.X < 0 || bounds.Y < 0 ||
                            bounds.Right > focus.Width || bounds.Bottom > focus.Height)
                            continue;

                        var sharpness = CandidateScorer.MeasureSharpnessContrast(focus, bounds);
                        var boundary = MeasureBoundary(edgeMap, bounds);

                        // Large candidates can contain smooth but still in-focus
                        // parts of a photograph. Do not discard them solely because
                        // a smaller subject has much higher local sharpness.
                        if (sharpness < 0.56 && areaRatio < 0.25)
                            continue;

                        var center = MeasureCenterPreference(bounds, focus.Width, focus.Height);
                        var score = sharpness * 0.72 + boundary * 0.18 + center * 0.10;
                        candidates.Add((bounds, score, areaRatio, widthRatio, heightRatio));
                    }
                }
            }
        }

        var results = new List<Rect>();
        foreach (var heightBand in candidates
                     .Where(candidate => candidate.WidthRatio >= 0.80)
                     .GroupBy(candidate => candidate.HeightRatio))
        {
            foreach (var candidate in heightBand.OrderByDescending(item => item.Score).Take(2))
                AddDistinctResult(results, candidate.Bounds, scale);
        }

        foreach (var areaBand in candidates
                     .GroupBy(candidate => GetAreaBand(candidate.AreaRatio))
                     .OrderBy(group => group.Key))
        {
            foreach (var candidate in areaBand.OrderByDescending(item => item.Score).Take(MaximumCandidatesPerAreaBand))
                AddDistinctResult(results, candidate.Bounds, scale);
        }

        foreach (var candidate in candidates.OrderByDescending(item => item.Score))
        {
            if (results.Count >= MaximumCandidates)
                break;

            AddDistinctResult(results, candidate.Bounds, scale);
        }

        return results;
    }

    private static int GetAreaBand(double areaRatio) => areaRatio switch
    {
        < 0.10 => 0,
        < 0.20 => 1,
        < 0.35 => 2,
        < 0.55 => 3,
        _ => 4
    };

    private static void AddDistinctResult(List<Rect> results, Rect bounds, double scale)
    {
        var originalBounds = DetectionGeometry.ScaleRect(bounds, 1.0 / scale);
        if (results.Count < MaximumCandidates &&
            results.All(existing => DetectionGeometry.IntersectionOverUnion(existing, originalBounds) < 0.78))
            results.Add(originalBounds);
    }

    private static double MeasureBoundary(Mat edges, Rect bounds)
    {
        var thickness = Math.Max(1, Math.Min(2, Math.Min(bounds.Width, bounds.Height) / 100));
        var top = DetectionGeometry.CountEdges(edges, new Rect(bounds.X, bounds.Y, bounds.Width, thickness)) /
                  (bounds.Width * (double)thickness);
        var bottom = DetectionGeometry.CountEdges(edges, new Rect(bounds.X, bounds.Bottom - thickness, bounds.Width, thickness)) /
                     (bounds.Width * (double)thickness);
        var left = DetectionGeometry.CountEdges(edges, new Rect(bounds.X, bounds.Y, thickness, bounds.Height)) /
                   (bounds.Height * (double)thickness);
        var right = DetectionGeometry.CountEdges(edges, new Rect(bounds.Right - thickness, bounds.Y, thickness, bounds.Height)) /
                    (bounds.Height * (double)thickness);

        return Math.Clamp(Math.Min(Math.Min(top, bottom), Math.Min(left, right)) * 4, 0, 1);
    }

    private static double MeasureCenterPreference(Rect bounds, int width, int height)
    {
        var x = (bounds.X + bounds.Width / 2.0 - width / 2.0) / (width / 2.0);
        var y = (bounds.Y + bounds.Height / 2.0 - height / 2.0) / (height / 2.0);
        return Math.Exp(-(x * x + y * y) / (2 * 0.45 * 0.45));
    }
}
