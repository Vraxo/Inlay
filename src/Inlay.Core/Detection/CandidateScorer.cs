using OpenCvSharp;

namespace Inlay.Detection;

internal static class CandidateScorer
{
    internal static double MeasureRectangleGeometry(Mat edges, Rect rect)
    {
        var thickness = Math.Max(1, Math.Min(3, Math.Min(rect.Width, rect.Height) / 100));
        var top = DetectionGeometry.CountEdges(edges, new Rect(rect.X, rect.Y, rect.Width, thickness)) /
                  (rect.Width * (double)thickness);
        var bottom = DetectionGeometry.CountEdges(edges, new Rect(rect.X, rect.Bottom - thickness, rect.Width, thickness)) /
                     (rect.Width * (double)thickness);
        var left = DetectionGeometry.CountEdges(edges, new Rect(rect.X, rect.Y, thickness, rect.Height)) /
                   (rect.Height * (double)thickness);
        var right = DetectionGeometry.CountEdges(edges, new Rect(rect.Right - thickness, rect.Y, thickness, rect.Height)) /
                    (rect.Height * (double)thickness);

        // A good rectangle needs all four sides, not just a strong average
        // that can hide one missing or misplaced edge.
        return Math.Clamp(Math.Min(Math.Min(top, bottom), Math.Min(left, right)) * 3.0, 0, 1);
    }

    internal static double MeasureBoundaryContinuity(Mat edges, Rect rect)
    {
        var thickness = Math.Max(1, Math.Min(3, Math.Min(rect.Width, rect.Height) / 100));
        var top = DetectionGeometry.CountEdges(edges, new Rect(rect.X, rect.Y, rect.Width, thickness));
        var bottom = DetectionGeometry.CountEdges(edges, new Rect(rect.X, rect.Bottom - thickness, rect.Width, thickness));
        var left = DetectionGeometry.CountEdges(edges, new Rect(rect.X, rect.Y, thickness, rect.Height));
        var right = DetectionGeometry.CountEdges(edges, new Rect(rect.Right - thickness, rect.Y, thickness, rect.Height));
        var horizontal = (top + bottom) / (2.0 * rect.Width * thickness);
        var vertical = (left + right) / (2.0 * rect.Height * thickness);
        return Math.Clamp((horizontal + vertical) * 0.5 * 2.2, 0, 1);
    }

    internal static double MeasureEdgeStrength(Mat gray, Rect rect)
    {
        using var region = new Mat(gray, DetectionGeometry.ClampRect(rect, gray.Width, gray.Height));
        using var gradientX = new Mat();
        using var gradientY = new Mat();
        using var magnitude = new Mat();
        Cv2.Sobel(region, gradientX, MatType.CV_32F, 1, 0);
        Cv2.Sobel(region, gradientY, MatType.CV_32F, 0, 1);
        Cv2.Magnitude(gradientX, gradientY, magnitude);
        return Math.Clamp(Cv2.Mean(magnitude).Val0 / 90.0, 0, 1);
    }

    internal static double MeasureSharpnessContrast(Mat focusMap, Rect rect)
    {
        var insetX = Math.Max(1, rect.Width / 10);
        var insetY = Math.Max(1, rect.Height / 10);
        var inner = DetectionGeometry.ClampRect(
            new Rect(rect.X + insetX, rect.Y + insetY,
                Math.Max(1, rect.Width - insetX * 2),
                Math.Max(1, rect.Height - insetY * 2)),
            focusMap.Width,
            focusMap.Height);

        var margin = Math.Max(8, Math.Min(rect.Width, rect.Height) / 16);
        var outer = DetectionGeometry.ClampRect(
            new Rect(rect.X - margin, rect.Y - margin,
                rect.Width + margin * 2, rect.Height + margin * 2),
            focusMap.Width,
            focusMap.Height);

        using var innerRegion = new Mat(focusMap, inner);
        using var outerRegion = new Mat(focusMap, outer);
        var innerArea = (double)inner.Width * inner.Height;
        var outerArea = (double)outer.Width * outer.Height;
        var ringArea = outerArea - innerArea;
        if (ringArea <= 0)
            return 0.5;

        var innerSharpness = Cv2.Mean(innerRegion).Val0;
        var outerMean = Cv2.Mean(outerRegion).Val0;
        var surroundingSharpness = Math.Max(0, (outerMean * outerArea - innerSharpness * innerArea) / ringArea);
        var contrast = (innerSharpness - surroundingSharpness) /
                       Math.Max(8.0, innerSharpness + surroundingSharpness);

        return Math.Clamp(0.5 + contrast * 1.4, 0, 1);
    }
}
