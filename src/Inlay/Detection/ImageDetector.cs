using OpenCvSharp;

namespace Inlay.Detection;

public sealed class ImageDetector
{
    private const int MaximumAnalysisDimension = 1600;

    public DetectionResult Detect(Mat source)
    {
        if (source.Empty() || source.Width < 80 || source.Height < 80)
            return new DetectionResult(DetectionStatus.Rejected, null, null, "Image is empty or too small.");

        var scale = Math.Min(1.0, (double)MaximumAnalysisDimension / Math.Max(source.Width, source.Height));
        using var analysis = new Mat();
        if (scale < 1.0)
            Cv2.Resize(source, analysis, new Size((int)Math.Round(source.Width * scale), (int)Math.Round(source.Height * scale)), interpolation: InterpolationFlags.Area);
        else
            source.CopyTo(analysis);

        using var gray = new Mat();
        using var blurred = new Mat();
        using var edges = new Mat();
        using var strongEdges = new Mat();
        using var softEdges = new Mat();
        using var closed = new Mat();
        using var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));
        Cv2.CvtColor(analysis, gray, ColorConversionCodes.BGR2GRAY);
        Cv2.GaussianBlur(gray, blurred, new Size(3, 3), 0);
        Cv2.Canny(blurred, strongEdges, 55, 145);
        Cv2.Canny(blurred, softEdges, 25, 75);
        Cv2.BitwiseOr(strongEdges, softEdges, edges);
        Cv2.MorphologyEx(edges, closed, MorphTypes.Close, kernel);

        Cv2.FindContours(closed, out Point[][] contours, out _, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
        var candidates = new List<DetectionCandidate>();
        var rejectedByBounds = 0;
        var rejectedByAspectRatio = 0;
        var rejectedByAreaRatio = 0;

        foreach (var contour in contours)
        {
            // Canny contours often trace only a thin edge fragment, so their
            // polygon area is not a reliable measure of the image region enclosed.
            // Use the bounding rectangle and its area for size filtering instead.
            var bounds = Cv2.BoundingRect(contour);
            if (bounds.Width < analysis.Width * 0.18 || bounds.Height < analysis.Height * 0.18 ||
                (bounds.Width >= analysis.Width * 0.985 && bounds.Height >= analysis.Height * 0.985))
            {
                rejectedByBounds++;
                continue;
            }

            var ratio = (double)bounds.Width / bounds.Height;
            if (ratio < 0.25 || ratio > 4.0)
            {
                rejectedByAspectRatio++;
                continue;
            }

            var polygon = Cv2.ApproxPolyDP(contour, Math.Max(2.0, Cv2.ArcLength(contour, true) * 0.018), true);
            var geometry = polygon.Length == 4 && Cv2.IsContourConvex(polygon) ? 1.0 : Math.Clamp(1.0 - Math.Abs(polygon.Length - 4) / 18.0, 0.35, 0.9);
            var areaRatio = (double)(bounds.Width * bounds.Height) / (analysis.Width * analysis.Height);
            if (areaRatio > 0.94 || areaRatio < 0.04)
            {
                rejectedByAreaRatio++;
                continue;
            }

            var boundary = MeasureBoundaryContinuity(edges, bounds);
            var edgeStrength = MeasureEdgeStrength(gray, bounds);
            var sharpness = MeasureSharpnessContrast(gray, bounds);
            var radius = EstimateCornerRadius(contour, bounds);
            var cornerFit = radius > 0 ? 0.85 : 0.65;

            var areaFit = Math.Clamp(1.0 - Math.Abs(areaRatio - 0.35) / 0.65, 0.25, 1.0);
            var confidence = boundary * 0.34 + edgeStrength * 0.18 + geometry * 0.18 +
                             sharpness * 0.18 + cornerFit * 0.07 + areaFit * 0.05;

            var candidate = new DetectionCandidate(
                ScaleRect(bounds, 1.0 / scale), confidence, boundary, edgeStrength,
                geometry, sharpness, cornerFit, (int)Math.Round(radius / scale), areaRatio);

            if (candidates.All(existing => IntersectionOverUnion(existing.Bounds, candidate.Bounds) < 0.82))
                candidates.Add(candidate);
            else
            {
                var duplicateIndex = candidates.FindIndex(existing => IntersectionOverUnion(existing.Bounds, candidate.Bounds) >= 0.82);
                if (duplicateIndex >= 0 && candidate.Confidence > candidates[duplicateIndex].Confidence)
                    candidates[duplicateIndex] = candidate;
            }
        }

        var ranked = candidates.OrderByDescending(x => x.Confidence).ToArray();
        if (ranked.Length == 0)
        {
            var reason = $"No candidate survived filtering. {contours.Length} contours examined: " +
                         $"{rejectedByBounds} outside size limits, " +
                         $"{rejectedByAspectRatio} outside aspect limits, {rejectedByAreaRatio} outside area limits.";
            return new DetectionResult(DetectionStatus.Rejected, null, null, reason);
        }

        // Rank the original contour candidates first. Refining every contour
        // changes their scores and can make an unrelated, vertically misplaced
        // rectangle win. Only adjust X after the best vertical candidate is chosen.
        var best = ranked[0];
        var analysisBounds = ScaleRect(best.Bounds, scale);
        var refinedAnalysisBounds = RefineHorizontalBounds(edges, analysisBounds);
        var refinedBounds = ScaleRect(refinedAnalysisBounds, 1.0 / scale);
        var finalBounds = new Rect(refinedBounds.X, best.Bounds.Y, refinedBounds.Width, best.Bounds.Height);

        // Confidence must describe the rectangle shown to the user, not the
        // original contour that was used to discover its vertical position.
        var finalBoundary = MeasureBoundaryContinuity(edges, refinedAnalysisBounds);
        var finalEdgeStrength = MeasureEdgeStrength(gray, refinedAnalysisBounds);
        var finalGeometry = MeasureRectangleGeometry(edges, refinedAnalysisBounds);
        var finalSharpness = MeasureSharpnessContrast(gray, refinedAnalysisBounds);
        var finalAreaRatio = (double)(refinedAnalysisBounds.Width * refinedAnalysisBounds.Height) /
                        (analysis.Width * analysis.Height);
        var finalAreaFit = Math.Clamp(1.0 - Math.Abs(finalAreaRatio - 0.35) / 0.65, 0.25, 1.0);
        var finalConfidence = finalBoundary * 0.34 + finalEdgeStrength * 0.18 + finalGeometry * 0.18 +
                              finalSharpness * 0.18 + best.CornerFit * 0.07 + finalAreaFit * 0.05;
        best = best with
        {
            Bounds = finalBounds,
            Confidence = finalConfidence,
            BoundaryContinuity = finalBoundary,
            EdgeStrength = finalEdgeStrength,
            GeometryFit = finalGeometry,
            SharpnessContrast = finalSharpness,
            AreaRatio = finalAreaRatio
        };

        var second = ranked.Length > 1 ? ranked[1] : null;
        var margin = second is null ? 1.0 : best.Confidence - second.Confidence;

        if (best.Confidence < DetectionResult.ReviewThreshold)
            return new DetectionResult(DetectionStatus.Rejected, best, second,
                $"Best candidate scored {best.Confidence:P0}, below the review threshold of {DetectionResult.ReviewThreshold:P0}.");

        if (best.Confidence < DetectionResult.AutomaticThreshold || margin < 0.08)
            return new DetectionResult(DetectionStatus.Review, best, second, margin < 0.08
                ? $"Top candidates are too close ({best.Confidence:P0} and {second!.Confidence:P0})."
                : $"Best candidate scored {best.Confidence:P0}; automatic extraction requires {DetectionResult.AutomaticThreshold:P0}.");

        return new DetectionResult(DetectionStatus.Extracted, best, second, "Candidate passed the initial heuristic thresholds.");
    }


    private static Rect RefineHorizontalBounds(Mat edges, Rect bounds)
    {
        var verticalInset = Math.Max(2, bounds.Height / 10);
        var verticalTop = bounds.Y + verticalInset;
        var verticalHeight = bounds.Height - verticalInset * 2;
        if (verticalHeight < 20 || bounds.Height <= 0)
            return bounds;

        var imageWidth = edges.Width;
        var imageHeight = edges.Height;
        using var verticalRegion = new Mat(edges, new Rect(0, verticalTop, imageWidth, verticalHeight));
        using var columnCounts = new Mat();
        Cv2.Reduce(verticalRegion, columnCounts, 0, ReduceTypes.Sum, MatType.CV_32S);

        var columnStrengths = new double[imageWidth];
        for (var x = 0; x < imageWidth; x++)
            columnStrengths[x] = columnCounts.At<int>(0, x) / (double)verticalHeight;

        var profile = new double[imageWidth];
        for (var x = 1; x < imageWidth - 1; x++)
            profile[x] = (columnStrengths[x - 1] + columnStrengths[x] + columnStrengths[x + 1]) / 3.0;

        var peaks = new List<(int X, double Strength)>();
        for (var x = 2; x < imageWidth - 2; x++)
        {
            if (profile[x] < 0.015 || profile[x] < profile[x - 1] || profile[x] < profile[x + 1])
                continue;

            if (peaks.All(peak => Math.Abs(peak.X - x) > 5))
                peaks.Add((x, profile[x]));
        }

        peaks = peaks.OrderByDescending(peak => peak.Strength).Take(64).ToList();
        if (peaks.Count < 2)
            return bounds;

        var thickness = Math.Max(1, Math.Min(3, bounds.Height / 100));
        var originalBoundary = MeasureBoundaryContinuity(edges, bounds);
        var widerCandidates = new List<(Rect Bounds, double Boundary)>();

        foreach (var left in peaks)
        {
            foreach (var right in peaks)
            {
                var width = right.X - left.X;
                var coversOriginal = left.X <= bounds.X && right.X >= bounds.Right - 6;
                if (!coversOriginal || width <= bounds.Width + 6 ||
                    width > bounds.Width * 2.5 || width < imageWidth * 0.18 ||
                    width >= imageWidth * 0.985)
                    continue;

                var ratio = (double)width / bounds.Height;
                var areaRatio = (double)width * bounds.Height / (imageWidth * imageHeight);
                if (ratio < 0.25 || ratio > 4.0 || areaRatio < 0.04 || areaRatio > 0.94)
                    continue;

                var horizontalInset = Math.Max(2, width / 10);
                var horizontalX = left.X + horizontalInset;
                var horizontalWidth = width - horizontalInset * 2;
                if (horizontalWidth <= 0)
                    continue;

                var leftStrength = left.Strength;
                var rightStrength = right.Strength;
                if (leftStrength < 0.015 || rightStrength < 0.015)
                    continue;

                var top = CountEdges(edges, new Rect(horizontalX, bounds.Y, horizontalWidth, thickness)) /
                          (horizontalWidth * (double)thickness);
                var bottom = CountEdges(edges, new Rect(horizontalX, bounds.Bottom - thickness, horizontalWidth, thickness)) /
                             (horizontalWidth * (double)thickness);
                if (top < 0.025 || bottom < 0.025)
                    continue;

                var continuity = (leftStrength + rightStrength + top + bottom) * 0.55;
                var verticalBalance = Math.Min(leftStrength, rightStrength) / Math.Max(leftStrength, rightStrength);
                var horizontalBalance = Math.Min(top, bottom) / Math.Max(top, bottom);
                continuity *= 0.75 + 0.25 * Math.Sqrt(verticalBalance * horizontalBalance);

                // A stronger internal line can produce a narrower but otherwise
                // plausible rectangle. Keep wider alternatives when all four
                // boundaries remain close in quality to the current candidate.
                if (continuity >= originalBoundary - 0.12)
                    widerCandidates.Add((new Rect(left.X, bounds.Y, width, bounds.Height), continuity));
            }
        }

        var wider = widerCandidates
            .OrderByDescending(candidate => candidate.Bounds.Width)
            .ThenByDescending(candidate => candidate.Boundary)
            .FirstOrDefault();

        return wider.Bounds.Width > 0 ? wider.Bounds : bounds;
    }

    private static double MeasureRectangleGeometry(Mat edges, Rect rect)
    {
        var thickness = Math.Max(1, Math.Min(3, Math.Min(rect.Width, rect.Height) / 100));
        var top = CountEdges(edges, new Rect(rect.X, rect.Y, rect.Width, thickness)) /
                  (rect.Width * (double)thickness);
        var bottom = CountEdges(edges, new Rect(rect.X, rect.Bottom - thickness, rect.Width, thickness)) /
                     (rect.Width * (double)thickness);
        var left = CountEdges(edges, new Rect(rect.X, rect.Y, thickness, rect.Height)) /
                   (rect.Height * (double)thickness);
        var right = CountEdges(edges, new Rect(rect.Right - thickness, rect.Y, thickness, rect.Height)) /
                    (rect.Height * (double)thickness);

        // A good rectangle needs all four sides, not just a strong average
        // that can hide one missing or misplaced edge.
        return Math.Clamp(Math.Min(Math.Min(top, bottom), Math.Min(left, right)) * 3.0, 0, 1);
    }

    private static double MeasureBoundaryContinuity(Mat edges, Rect rect)
    {
        var thickness = Math.Max(1, Math.Min(3, Math.Min(rect.Width, rect.Height) / 100));
        var top = CountEdges(edges, new Rect(rect.X, rect.Y, rect.Width, thickness));
        var bottom = CountEdges(edges, new Rect(rect.X, rect.Bottom - thickness, rect.Width, thickness));
        var left = CountEdges(edges, new Rect(rect.X, rect.Y, thickness, rect.Height));
        var right = CountEdges(edges, new Rect(rect.Right - thickness, rect.Y, thickness, rect.Height));
        var horizontal = (top + bottom) / (2.0 * rect.Width * thickness);
        var vertical = (left + right) / (2.0 * rect.Height * thickness);
        return Math.Clamp((horizontal + vertical) * 0.5 * 2.2, 0, 1);
    }

    private static double CountEdges(Mat edges, Rect rect)
    {
        rect = ClampRect(rect, edges.Width, edges.Height);
        if (rect.Width <= 0 || rect.Height <= 0)
            return 0;
        using var region = new Mat(edges, rect);
        return Cv2.CountNonZero(region);
    }

    private static double MeasureEdgeStrength(Mat gray, Rect rect)
    {
        using var region = new Mat(gray, ClampRect(rect, gray.Width, gray.Height));
        using var gradientX = new Mat();
        using var gradientY = new Mat();
        using var magnitude = new Mat();
        Cv2.Sobel(region, gradientX, MatType.CV_32F, 1, 0);
        Cv2.Sobel(region, gradientY, MatType.CV_32F, 0, 1);
        Cv2.Magnitude(gradientX, gradientY, magnitude);
        return Math.Clamp(Cv2.Mean(magnitude).Val0 / 90.0, 0, 1);
    }

    private static double MeasureSharpnessContrast(Mat gray, Rect rect)
    {
        var inner = ClampRect(new Rect(rect.X + rect.Width / 8, rect.Y + rect.Height / 8, rect.Width * 3 / 4, rect.Height * 3 / 4), gray.Width, gray.Height);
        var outer = ClampRect(new Rect(Math.Max(0, rect.X - 12), Math.Max(0, rect.Y - 12),
            Math.Min(gray.Width - Math.Max(0, rect.X - 12), rect.Width + 24),
            Math.Min(gray.Height - Math.Max(0, rect.Y - 12), rect.Height + 24)), gray.Width, gray.Height);
        using var innerMat = new Mat(gray, inner);
        using var innerLap = new Mat();
        Cv2.Laplacian(innerMat, innerLap, MatType.CV_32F);
        Cv2.MeanStdDev(innerLap, out _, out var innerStd);
        var innerVariance = innerStd.Val0 * innerStd.Val0;

        using var outerMat = new Mat(gray, outer);
        using var outerLap = new Mat();
        Cv2.Laplacian(outerMat, outerLap, MatType.CV_32F);
        Cv2.MeanStdDev(outerLap, out _, out var outerStd);
        var outerVariance = outerStd.Val0 * outerStd.Val0;
        var contrast = (innerVariance - outerVariance) / Math.Max(1.0, innerVariance + outerVariance);
        return Math.Clamp(0.5 + contrast * 0.9, 0, 1);
    }

    private static double EstimateCornerRadius(Point[] contour, Rect bounds)
    {
        var radii = new List<double>(4);
        var marginX = bounds.Width * 0.28;
        var marginY = bounds.Height * 0.28;
        foreach (var (cornerX, cornerY, left, top) in new[]
        {
            (bounds.Left, bounds.Top, true, true),
            (bounds.Right - 1, bounds.Top, false, true),
            (bounds.Left, bounds.Bottom - 1, true, false),
            (bounds.Right - 1, bounds.Bottom - 1, false, false)
        })
        {
            var points = contour.Where(p =>
                (left ? p.X <= bounds.Left + marginX : p.X >= bounds.Right - marginX) &&
                (top ? p.Y <= bounds.Top + marginY : p.Y >= bounds.Bottom - marginY));
            var distance = points.Select(p => Math.Sqrt(Math.Pow(p.X - cornerX, 2) + Math.Pow(p.Y - cornerY, 2)))
                .DefaultIfEmpty(0).Min();
            radii.Add(distance);
        }

        var positive = radii.Where(x => x >= 3).ToArray();
        if (positive.Length < 3)
            return 0;
        var median = positive.Order().ElementAt(positive.Length / 2);
        var deviation = positive.Max(x => Math.Abs(x - median));
        return deviation <= Math.Max(5, median * 0.35) ? Math.Clamp(median, 0, Math.Min(bounds.Width, bounds.Height) * 0.15) : 0;
    }

    private static Rect ScaleRect(Rect rect, double scale) => new(
        (int)Math.Round(rect.X * scale), (int)Math.Round(rect.Y * scale),
        (int)Math.Round(rect.Width * scale), (int)Math.Round(rect.Height * scale));

    private static Rect ClampRect(Rect rect, int width, int height)
    {
        var x = Math.Clamp(rect.X, 0, width - 1);
        var y = Math.Clamp(rect.Y, 0, height - 1);
        return new Rect(x, y, Math.Clamp(rect.Width, 1, width - x), Math.Clamp(rect.Height, 1, height - y));
    }

    private static double IntersectionOverUnion(Rect a, Rect b)
    {
        var left = Math.Max(a.Left, b.Left);
        var top = Math.Max(a.Top, b.Top);
        var right = Math.Min(a.Right, b.Right);
        var bottom = Math.Min(a.Bottom, b.Bottom);
        var intersectionWidth = Math.Max(0, right - left);
        var intersectionHeight = Math.Max(0, bottom - top);
        var intersectionArea = (double)intersectionWidth * intersectionHeight;
        var unionArea = (double)a.Width * a.Height + (double)b.Width * b.Height - intersectionArea;
        return unionArea <= 0 ? 0 : intersectionArea / unionArea;
    }
}
