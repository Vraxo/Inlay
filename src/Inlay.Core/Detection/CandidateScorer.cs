using OpenCvSharp;

namespace Inlay.Detection;

internal static class CandidateScorer
{
    internal static double MeasureCenterPreference(Rect bounds, int imageWidth, int imageHeight)
    {
        var offsetX = (bounds.X + bounds.Width / 2.0 - imageWidth / 2.0) / (imageWidth / 2.0);
        var offsetY = (bounds.Y + bounds.Height / 2.0 - imageHeight / 2.0) / (imageHeight / 2.0);
        var distance = Math.Sqrt(offsetX * offsetX + offsetY * offsetY);

        // A soft prior: centered candidates receive a modest boost, while
        // plausible off-center candidates remain eligible.
        const double falloff = 0.55;
        return Math.Exp(-(distance * distance) / (2 * falloff * falloff));
    }

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
        rect = DetectionGeometry.ClampRect(rect, focusMap.Width, focusMap.Height);
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

        var innerSharpness = Mean(focusMap, inner);
        var surroundingSharpness = MeasureOutsideMean(focusMap, rect, outer);
        var contrast = (innerSharpness - surroundingSharpness) /
                       Math.Max(8.0, innerSharpness + surroundingSharpness);

        // Keep the full contrast range. A steeper multiplier saturates both
        // small detailed patches and the complete foreground image at 1.0.
        return Math.Clamp(0.5 + contrast * 0.5, 0, 1);
    }

    private static double MeasureOutsideMean(Mat image, Rect bounds, Rect outer)
    {
        var regions = new[]
        {
            bounds.Y > outer.Y
                ? new Rect(outer.X, outer.Y, outer.Width, bounds.Y - outer.Y)
                : default,
            bounds.Bottom < outer.Bottom
                ? new Rect(outer.X, bounds.Bottom, outer.Width, outer.Bottom - bounds.Bottom)
                : default,
            bounds.X > outer.X
                ? new Rect(outer.X, bounds.Y, bounds.X - outer.X, bounds.Height)
                : default,
            bounds.Right < outer.Right
                ? new Rect(bounds.Right, bounds.Y, outer.Right - bounds.Right, bounds.Height)
                : default
        };

        double weightedSum = 0;
        var totalArea = 0;
        foreach (var region in regions)
        {
            var clipped = ClipToImage(region, image.Width, image.Height);
            if (clipped is not Rect validRegion)
                continue;

            var area = validRegion.Width * validRegion.Height;
            weightedSum += Mean(image, validRegion) * area;
            totalArea += area;
        }

        return totalArea == 0 ? 0 : weightedSum / totalArea;
    }

    private static Rect? ClipToImage(Rect bounds, int width, int height)
    {
        var left = Math.Clamp(bounds.X, 0, width);
        var top = Math.Clamp(bounds.Y, 0, height);
        var right = Math.Clamp(bounds.Right, 0, width);
        var bottom = Math.Clamp(bounds.Bottom, 0, height);
        return right <= left || bottom <= top
            ? null
            : new Rect(left, top, right - left, bottom - top);
    }

    private static double Mean(Mat image, Rect bounds)
    {
        using var region = new Mat(image, bounds);
        return Cv2.Mean(region).Val0;
    }
}
