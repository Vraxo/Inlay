using OpenCvSharp;

namespace Inlay.Detection;

internal static class CenteredRectangleDetector
{
    private const int MaximumSearchDimension = 160;

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

        var candidates = new List<(Rect Bounds, double Score)>();
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

                        var sharpness = MeasureSharpnessSeparation(focus, bounds);
                        if (sharpness < 0.56)
                            continue;

                        var boundary = MeasureBoundary(edgeMap, bounds);
                        var center = MeasureCenterPreference(bounds, focus.Width, focus.Height);
                        var score = sharpness * 0.72 + boundary * 0.18 + center * 0.10;
                        candidates.Add((bounds, score));
                    }
                }
            }
        }

        var results = new List<Rect>();
        foreach (var candidate in candidates.OrderByDescending(item => item.Score))
        {
            var originalBounds = DetectionGeometry.ScaleRect(candidate.Bounds, 1.0 / scale);
            if (results.All(existing => DetectionGeometry.IntersectionOverUnion(existing, originalBounds) < 0.78))
                results.Add(originalBounds);

            if (results.Count == 20)
                break;
        }

        return results;
    }

    private static double MeasureSharpnessSeparation(Mat focus, Rect bounds)
    {
        var insetX = Math.Max(1, bounds.Width / 12);
        var insetY = Math.Max(1, bounds.Height / 12);
        var inner = new Rect(
            bounds.X + insetX,
            bounds.Y + insetY,
            bounds.Width - insetX * 2,
            bounds.Height - insetY * 2);
        var marginX = Math.Max(2, bounds.Width / 18);
        var marginY = Math.Max(2, bounds.Height / 18);
        var outer = DetectionGeometry.ClampRect(
            new Rect(bounds.X - marginX, bounds.Y - marginY,
                bounds.Width + marginX * 2, bounds.Height + marginY * 2),
            focus.Width,
            focus.Height);

        var insideMean = Mean(focus, inner);
        var outsideMean = Mean(focus, outer);
        var innerArea = (double)inner.Width * inner.Height;
        var outerArea = (double)outer.Width * outer.Height;
        var ringArea = outerArea - innerArea;
        if (ringArea <= 0)
            return 0.5;

        var ringMean = Math.Max(0, (outsideMean * outerArea - insideMean * innerArea) / ringArea);
        var contrast = (insideMean - ringMean) / Math.Max(8, insideMean + ringMean);
        return Math.Clamp(0.5 + contrast * 1.5, 0, 1);
    }

    private static double Mean(Mat image, Rect bounds)
    {
        double sum = 0;
        var samples = 0;
        for (var y = bounds.Y; y < bounds.Bottom; y += 2)
        {
            for (var x = bounds.X; x < bounds.Right; x += 2)
            {
                sum += image.At<byte>(y, x);
                samples++;
            }
        }

        return sum / Math.Max(1, samples);
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
