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
            Cv2.Resize(source, analysis,
                new Size((int)Math.Round(source.Width * scale), (int)Math.Round(source.Height * scale)),
                interpolation: InterpolationFlags.Area);
        else
            source.CopyTo(analysis);

        using var gray = new Mat();
        using var blurred = new Mat();
        using var edges = new Mat();
        using var strongEdges = new Mat();
        using var softEdges = new Mat();
        using var closed = new Mat();
        using var focusMap = new Mat();
        using var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));

        Cv2.CvtColor(analysis, gray, ColorConversionCodes.BGR2GRAY);
        Cv2.GaussianBlur(gray, blurred, new Size(3, 3), 0);
        Cv2.Canny(blurred, strongEdges, 55, 145);
        Cv2.Canny(blurred, softEdges, 25, 75);
        Cv2.BitwiseOr(strongEdges, softEdges, edges);
        Cv2.MorphologyEx(edges, closed, MorphTypes.Close, kernel);
        FocusRegionDetector.CreateFocusMap(gray, focusMap);

        Cv2.FindContours(closed, out Point[][] contours, out _, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
        var candidates = new List<DetectionCandidate>();
        var rejectedByBounds = 0;
        var rejectedByAspectRatio = 0;
        var rejectedByAreaRatio = 0;

        foreach (var contour in contours)
        {
            var bounds = Cv2.BoundingRect(contour);
            if (!HasCandidateSize(bounds, analysis.Width, analysis.Height))
            {
                rejectedByBounds++;
                continue;
            }

            var ratio = (double)bounds.Width / bounds.Height;
            if (ratio is < 0.25 or > 4.0)
            {
                rejectedByAspectRatio++;
                continue;
            }

            var polygon = Cv2.ApproxPolyDP(contour, Math.Max(2.0, Cv2.ArcLength(contour, true) * 0.018), true);
            var geometry = polygon.Length == 4 && Cv2.IsContourConvex(polygon)
                ? 1.0
                : Math.Clamp(1.0 - Math.Abs(polygon.Length - 4) / 18.0, 0.35, 0.9);

            AddCandidate(candidates, edges, gray, focusMap, bounds, geometry, scale,
                analysis.Width, analysis.Height, ref rejectedByAreaRatio);
        }

        var focusCandidates = FocusRegionDetector.FindCandidates(focusMap);
        foreach (var bounds in focusCandidates)
            AddCandidate(candidates, edges, gray, focusMap, bounds, 0, scale,
                analysis.Width, analysis.Height, ref rejectedByAreaRatio);

        var ranked = candidates.OrderByDescending(candidate => candidate.Confidence).ToArray();
        if (ranked.Length == 0)
        {
            var reason = $"No candidate survived filtering. {contours.Length} edge contours and " +
                         $"{focusCandidates.Count} focus regions examined: {rejectedByBounds} outside size limits, " +
                         $"{rejectedByAspectRatio} outside aspect limits, {rejectedByAreaRatio} outside area limits.";
            return new DetectionResult(DetectionStatus.Rejected, null, null, reason);
        }

        var best = ranked[0];
        var analysisBounds = DetectionGeometry.ScaleRect(best.Bounds, scale);
        var refinedAnalysisBounds = CropRefiner.RefineHorizontalBounds(edges, analysisBounds);
        var refinedBounds = DetectionGeometry.ScaleRect(refinedAnalysisBounds, 1.0 / scale);
        var finalBounds = new Rect(refinedBounds.X, best.Bounds.Y, refinedBounds.Width, best.Bounds.Height);

        using var sourceGray = new Mat();
        using var sourceBlurred = new Mat();
        using var sourceStrongEdges = new Mat();
        using var sourceSoftEdges = new Mat();
        using var sourceEdges = new Mat();
        Cv2.CvtColor(source, sourceGray, ColorConversionCodes.BGR2GRAY);
        Cv2.GaussianBlur(sourceGray, sourceBlurred, new Size(3, 3), 0);
        Cv2.Canny(sourceBlurred, sourceStrongEdges, 55, 145);
        Cv2.Canny(sourceBlurred, sourceSoftEdges, 25, 75);
        Cv2.BitwiseOr(sourceStrongEdges, sourceSoftEdges, sourceEdges);
        finalBounds = CropRefiner.RefineTopBoundary(sourceEdges, finalBounds, 4);
        finalBounds = CropRefiner.RefineBottomBoundary(sourceEdges, finalBounds, 6);

        var finalCornerRadius = CornerRadiusEstimator.EstimateCornerRadius(sourceEdges, finalBounds);
        var finalCornerFit = finalCornerRadius > 0 ? 0.85 : 0.65;
        var finalBoundary = CandidateScorer.MeasureBoundaryContinuity(edges, refinedAnalysisBounds);
        var finalEdgeStrength = CandidateScorer.MeasureEdgeStrength(gray, refinedAnalysisBounds);
        var finalGeometry = CandidateScorer.MeasureRectangleGeometry(edges, refinedAnalysisBounds);
        var finalSharpness = CandidateScorer.MeasureSharpnessContrast(focusMap, refinedAnalysisBounds);
        var finalConfidence = ScoreCandidate(
            finalBoundary, finalEdgeStrength, finalGeometry, finalSharpness, finalCornerFit);

        best = best with
        {
            Bounds = finalBounds,
            Confidence = finalConfidence,
            BoundaryContinuity = finalBoundary,
            EdgeStrength = finalEdgeStrength,
            GeometryFit = finalGeometry,
            SharpnessContrast = finalSharpness,
            CornerFit = finalCornerFit,
            CornerRadius = finalCornerRadius,
            AreaRatio = (double)refinedAnalysisBounds.Width * refinedAnalysisBounds.Height /
                        (analysis.Width * analysis.Height)
        };

        var second = ranked.Length > 1 ? ranked[1] : null;
        var margin = second is null ? 1.0 : best.Confidence - second.Confidence;
        if (best.Confidence < DetectionResult.ReviewThreshold && (second is null || margin >= 0.08))
            return new DetectionResult(DetectionStatus.Rejected, best, second,
                "A possible foreground region was found, but its evidence is too weak for a reliable crop.");
        if (best.Confidence < DetectionResult.AutomaticThreshold || margin < 0.08)
            return new DetectionResult(DetectionStatus.Review, best, second,
                "A foreground region was found, but its boundaries or separation are uncertain.");

        return new DetectionResult(DetectionStatus.Extracted, best, second,
            "Foreground boundaries and focus separation passed the initial heuristic thresholds.");
    }

    private static void AddCandidate(
        List<DetectionCandidate> candidates,
        Mat edges,
        Mat gray,
        Mat focusMap,
        Rect bounds,
        double geometryHint,
        double scale,
        int imageWidth,
        int imageHeight,
        ref int rejectedByAreaRatio)
    {
        var areaRatio = (double)bounds.Width * bounds.Height / (imageWidth * imageHeight);
        if (areaRatio is < 0.04 or > 0.94)
        {
            rejectedByAreaRatio++;
            return;
        }

        var boundary = CandidateScorer.MeasureBoundaryContinuity(edges, bounds);
        var edgeStrength = CandidateScorer.MeasureEdgeStrength(gray, bounds);
        var sharpness = CandidateScorer.MeasureSharpnessContrast(focusMap, bounds);
        var geometry = Math.Max(geometryHint, CandidateScorer.MeasureRectangleGeometry(edges, bounds));
        const double cornerFit = 0.65;
        var confidence = ScoreCandidate(boundary, edgeStrength, geometry, sharpness, cornerFit);
        var candidate = new DetectionCandidate(
            DetectionGeometry.ScaleRect(bounds, 1.0 / scale), confidence, boundary, edgeStrength,
            geometry, sharpness, cornerFit, 0, areaRatio);

        var duplicateIndex = candidates.FindIndex(existing =>
            DetectionGeometry.IntersectionOverUnion(existing.Bounds, candidate.Bounds) >= 0.82);
        if (duplicateIndex < 0)
            candidates.Add(candidate);
        else if (candidate.Confidence > candidates[duplicateIndex].Confidence)
            candidates[duplicateIndex] = candidate;
    }

    private static bool HasCandidateSize(Rect bounds, int imageWidth, int imageHeight) =>
        bounds.Width >= imageWidth * 0.18 &&
        bounds.Height >= imageHeight * 0.12;

    private static double ScoreCandidate(
        double boundary,
        double edgeStrength,
        double geometry,
        double sharpness,
        double cornerFit) =>
        boundary * 0.30 +
        edgeStrength * 0.12 +
        geometry * 0.20 +
        sharpness * 0.33 +
        cornerFit * 0.05;
}
