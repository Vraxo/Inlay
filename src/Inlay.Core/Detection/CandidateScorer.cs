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
        rect = DetectionGeometry.ClampRect(rect, edges.Width, edges.Height);
        var thickness = GetBoundaryThickness(rect);
        var tolerance = GetBoundaryTolerance(rect);
        var top = MeasureBestSideCoverage(edges, rect, thickness, tolerance, true, true);
        var bottom = MeasureBestSideCoverage(edges, rect, thickness, tolerance, true, false);
        var left = MeasureBestSideCoverage(edges, rect, thickness, tolerance, false, true);
        var right = MeasureBestSideCoverage(edges, rect, thickness, tolerance, false, false);

        // A good rectangle needs all four sides. A few pixels of alignment
        // error should not make an otherwise supported side disappear.
        return Math.Clamp(Math.Min(Math.Min(top, bottom), Math.Min(left, right)) * 3.0, 0, 1);
    }

    internal static double MeasureBoundaryContinuity(Mat edges, Rect rect)
    {
        rect = DetectionGeometry.ClampRect(rect, edges.Width, edges.Height);
        var thickness = GetBoundaryThickness(rect);
        var tolerance = GetBoundaryTolerance(rect);
        var top = MeasureBestSideCoverage(edges, rect, thickness, tolerance, true, true);
        var bottom = MeasureBestSideCoverage(edges, rect, thickness, tolerance, true, false);
        var left = MeasureBestSideCoverage(edges, rect, thickness, tolerance, false, true);
        var right = MeasureBestSideCoverage(edges, rect, thickness, tolerance, false, false);
        var horizontal = (top + bottom) * 0.5;
        var vertical = (left + right) * 0.5;
        return Math.Clamp((horizontal + vertical) * 0.5 * 2.2, 0, 1);
    }

    internal static double MeasureBoundaryContrast(Mat gray, Rect rect)
    {
        rect = DetectionGeometry.ClampRect(rect, gray.Width, gray.Height);
        var thickness = GetBoundaryThickness(rect);
        var tolerance = GetBoundaryTolerance(rect);
        var top = MeasureBestBoundaryContrast(gray, rect, thickness, tolerance, true, true);
        var bottom = MeasureBestBoundaryContrast(gray, rect, thickness, tolerance, true, false);
        var left = MeasureBestBoundaryContrast(gray, rect, thickness, tolerance, false, true);
        var right = MeasureBestBoundaryContrast(gray, rect, thickness, tolerance, false, false);

        // Require contrast on every side so a single strong edge cannot
        // make an unrelated rectangle appear reliable.
        return Math.Clamp(Math.Min(Math.Min(top, bottom), Math.Min(left, right)) / 64.0, 0, 1);
    }

    private static double MeasureBestSideCoverage(
        Mat edges,
        Rect bounds,
        int thickness,
        int tolerance,
        bool horizontal,
        bool leading)
    {
        var best = 0.0;
        for (var offset = -tolerance; offset <= tolerance; offset++)
        {
            var region = horizontal
                ? leading
                    ? new Rect(bounds.X, bounds.Y + offset, bounds.Width, thickness)
                    : new Rect(bounds.X, bounds.Bottom - thickness + offset, bounds.Width, thickness)
                : leading
                    ? new Rect(bounds.X + offset, bounds.Y, thickness, bounds.Height)
                    : new Rect(bounds.Right - thickness + offset, bounds.Y, thickness, bounds.Height);

            if (!IsInsideImage(edges, region))
                continue;

            var area = region.Width * (double)region.Height;
            best = Math.Max(best, DetectionGeometry.CountEdges(edges, region) / area);
        }

        return best;
    }

    private static double MeasureBestBoundaryContrast(
        Mat gray,
        Rect bounds,
        int thickness,
        int tolerance,
        bool horizontal,
        bool leading)
    {
        var inset = Math.Max(2, (horizontal ? bounds.Width : bounds.Height) / 10);
        var span = (horizontal ? bounds.Width : bounds.Height) - inset * 2;
        if (span <= 0)
            return 0;

        var best = 0.0;
        for (var offset = -tolerance; offset <= tolerance; offset++)
        {
            Rect inside;
            Rect outside;
            if (horizontal)
            {
                var boundary = (leading ? bounds.Y : bounds.Bottom) + offset;
                inside = leading
                    ? new Rect(bounds.X + inset, boundary, span, thickness)
                    : new Rect(bounds.X + inset, boundary - thickness, span, thickness);
                outside = leading
                    ? new Rect(bounds.X + inset, boundary - thickness, span, thickness)
                    : new Rect(bounds.X + inset, boundary, span, thickness);
            }
            else
            {
                var boundary = (leading ? bounds.X : bounds.Right) + offset;
                inside = leading
                    ? new Rect(boundary, bounds.Y + inset, thickness, span)
                    : new Rect(boundary - thickness, bounds.Y + inset, thickness, span);
                outside = leading
                    ? new Rect(boundary - thickness, bounds.Y + inset, thickness, span)
                    : new Rect(boundary, bounds.Y + inset, thickness, span);
            }

            if (!IsInsideImage(gray, inside) || !IsInsideImage(gray, outside))
                continue;

            using var insideRegion = new Mat(gray, inside);
            using var outsideRegion = new Mat(gray, outside);
            using var difference = new Mat();
            Cv2.Absdiff(insideRegion, outsideRegion, difference);
            best = Math.Max(best, Cv2.Mean(difference).Val0);
        }

        return best;
    }

    private static int GetBoundaryThickness(Rect rect) =>
        Math.Max(1, Math.Min(3, Math.Min(rect.Width, rect.Height) / 100));

    private static int GetBoundaryTolerance(Rect rect) =>
        Math.Clamp(Math.Min(rect.Width, rect.Height) / 160, 2, 6);

    private static bool IsInsideImage(Mat image, Rect rect) =>
        rect.X >= 0 &&
        rect.Y >= 0 &&
        rect.Right <= image.Width &&
        rect.Bottom <= image.Height;

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
