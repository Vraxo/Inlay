using OpenCvSharp;

namespace Inlay.Detection;

public sealed class ImageDetector
{
    private const int MaximumAnalysisDimension = 1600;
    private const int SquareCornerHorizontalInset = 3;

    public DetectionResult Detect(Mat source) => DetectAsync(source).GetAwaiter().GetResult();

    public async Task<DetectionResult> DetectAsync(
        Mat source,
        Action<int>? reportProgress = null,
        Func<Task>? yieldToBrowser = null)
    {
        if (source.Empty() || source.Width < 80 || source.Height < 80)
            return new DetectionResult(DetectionStatus.Rejected, null, null, "Image is empty or too small.");

        await ReportProgress(2, reportProgress, yieldToBrowser);
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

        await ReportProgress(5, reportProgress, yieldToBrowser);
        Cv2.CvtColor(analysis, gray, ColorConversionCodes.BGR2GRAY);
        await ReportProgress(8, reportProgress, yieldToBrowser);
        Cv2.GaussianBlur(gray, blurred, new Size(3, 3), 0);
        await ReportProgress(11, reportProgress, yieldToBrowser);
        Cv2.Canny(blurred, strongEdges, 55, 145);
        await ReportProgress(14, reportProgress, yieldToBrowser);
        Cv2.Canny(blurred, softEdges, 25, 75);
        await ReportProgress(16, reportProgress, yieldToBrowser);
        Cv2.BitwiseOr(strongEdges, softEdges, edges);
        Cv2.MorphologyEx(edges, closed, MorphTypes.Close, kernel);
        await ReportProgress(18, reportProgress, yieldToBrowser);
        FocusRegionDetector.CreateFocusMap(gray, focusMap);
        await ReportProgress(28, reportProgress, yieldToBrowser);

        Cv2.FindContours(closed, out Point[][] contours, out _, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
        var candidates = new List<RankedCandidate>();
        var rejectedByBounds = 0;
        var rejectedByAspectRatio = 0;
        var rejectedByAreaRatio = 0;

        var contourIndex = 0;
        foreach (var contour in contours)
        {
            if ((contourIndex++ & 7) == 0)
            {
                reportProgress?.Invoke(28 + Math.Min(12, contourIndex * 12 / Math.Max(1, contours.Length)));
                await YieldToBrowser(yieldToBrowser);
            }
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

            AddCandidate(candidates, edges, gray, focusMap, bounds, geometry, false, scale,
                analysis.Width, analysis.Height, ref rejectedByAreaRatio);
        }

        await ReportProgress(42, reportProgress, yieldToBrowser);
        var focusCandidates = await FocusRegionDetector.FindCandidatesAsync(focusMap, yieldToBrowser);
        await ReportProgress(52, reportProgress, yieldToBrowser);
        var focusCandidateIndex = 0;
        foreach (var bounds in focusCandidates)
        {
            if ((focusCandidateIndex++ & 3) == 0)
                await YieldToBrowser(yieldToBrowser);
            AddCandidate(candidates, edges, gray, focusMap, bounds, 0, true, scale,
                analysis.Width, analysis.Height, ref rejectedByAreaRatio);
        }

        var rectangleCandidates = await CenteredRectangleDetector.FindCandidatesAsync(focusMap, edges, yieldToBrowser,
            progress => reportProgress?.Invoke(52 + progress * 18 / 100));
        await ReportProgress(70, reportProgress, yieldToBrowser);
        var rectangleCandidateIndex = 0;
        foreach (var bounds in rectangleCandidates)
        {
            if ((rectangleCandidateIndex++ & 3) == 0)
                await YieldToBrowser(yieldToBrowser);
            AddCandidate(candidates, edges, gray, focusMap, bounds, 0, true, scale,
                analysis.Width, analysis.Height, ref rejectedByAreaRatio);
        }

        var ranked = candidates.OrderByDescending(candidate => candidate.Result.Confidence).ToArray();
        if (ranked.Length == 0)
        {
            var reason = $"No candidate survived filtering. {contours.Length} edge contours and " +
                         $"{focusCandidates.Count} focus regions examined: {rejectedByBounds} outside size limits, " +
                         $"{rejectedByAspectRatio} outside aspect limits, {rejectedByAreaRatio} outside area limits.";
            return new DetectionResult(DetectionStatus.Rejected, null, null, reason);
        }

        var highestConfidence = ranked[0].Result.Confidence;
        var largerAlternative = ranked
            .Where(candidate =>
                ranked[0].Result.AreaRatio < 0.15 &&
                candidate.Result.AreaRatio >= Math.Max(0.25, ranked[0].Result.AreaRatio * 2.0) &&
                candidate.Result.Bounds.Width >= source.Width * 0.90 &&
                highestConfidence - candidate.Result.Confidence < 0.16 &&
                CandidateScorer.MeasureCenterPreference(
                    candidate.Result.Bounds, source.Width, source.Height) >= 0.75)
            .OrderByDescending(candidate => candidate.Result.Confidence)
            .ThenBy(candidate => candidate.Result.AreaRatio)
            .FirstOrDefault();

        // When two centered candidates are nearly tied, prefer the more complete
        // rectangle. This prevents a detailed subject inside a larger image from
        // winning solely because its local sharpness is higher.
        var bestMatch = largerAlternative ?? ranked[0];
        var best = bestMatch.Result;
        var analysisBounds = DetectionGeometry.ScaleRect(best.Bounds, scale);
        var hasUsableBoundaries = best.BoundaryContinuity >= 0.12;
        var refinedAnalysisBounds = hasUsableBoundaries
            ? CropRefiner.RefineHorizontalBounds(edges, analysisBounds)
            : analysisBounds;
        var refinedBounds = DetectionGeometry.ScaleRect(refinedAnalysisBounds, 1.0 / scale);
        var finalBounds = new Rect(refinedBounds.X, best.Bounds.Y, refinedBounds.Width, best.Bounds.Height);

        using var sourceGray = new Mat();
        using var sourceBlurred = new Mat();
        using var sourceStrongEdges = new Mat();
        using var sourceSoftEdges = new Mat();
        using var sourceEdges = new Mat();
        using var sourceFocusMap = new Mat();
        await ReportProgress(78, reportProgress, yieldToBrowser);
        Cv2.CvtColor(source, sourceGray, ColorConversionCodes.BGR2GRAY);
        await ReportProgress(80, reportProgress, yieldToBrowser);
        Cv2.GaussianBlur(sourceGray, sourceBlurred, new Size(3, 3), 0);
        await ReportProgress(82, reportProgress, yieldToBrowser);
        Cv2.Canny(sourceBlurred, sourceStrongEdges, 55, 145);
        await ReportProgress(84, reportProgress, yieldToBrowser);
        Cv2.Canny(sourceBlurred, sourceSoftEdges, 25, 75);
        await ReportProgress(86, reportProgress, yieldToBrowser);
        Cv2.BitwiseOr(sourceStrongEdges, sourceSoftEdges, sourceEdges);
        await ReportProgress(88, reportProgress, yieldToBrowser);
        FocusRegionDetector.CreateFocusMap(sourceGray, sourceFocusMap);
        await ReportProgress(90, reportProgress, yieldToBrowser);
        var finalCornerRadius = 0;
        if (hasUsableBoundaries)
        {
            await YieldToBrowser(yieldToBrowser);
            finalBounds = CropRefiner.RefineTopBoundary(sourceEdges, sourceFocusMap, finalBounds, 120);
            await ReportProgress(92, reportProgress, yieldToBrowser);
            finalBounds = CropRefiner.RefineBottomBoundary(sourceEdges, finalBounds, 120);
            await ReportProgress(94, reportProgress, yieldToBrowser);
            finalCornerRadius = CornerRadiusEstimator.EstimateCornerRadius(sourceEdges, finalBounds);
        }

        // Square-cornered screenshots can retain a thin antialiased halo
        // outside the visible image edge. Remove the small horizontal fringe
        // only when all four corners were classified as square.
        if (finalCornerRadius == 0 && hasUsableBoundaries && finalBounds.Width > SquareCornerHorizontalInset * 2)
        {
            finalBounds = new Rect(
                finalBounds.X + SquareCornerHorizontalInset,
                finalBounds.Y,
                finalBounds.Width - SquareCornerHorizontalInset * 2,
                finalBounds.Height);
        }

        await ReportProgress(96, reportProgress, yieldToBrowser);
        var finalCornerFit = finalCornerRadius > 0 ? 0.85 : 0.65;
        var finalBoundary = CandidateScorer.MeasureBoundaryContinuity(edges, refinedAnalysisBounds);
        var finalEdgeStrength = CandidateScorer.MeasureEdgeStrength(gray, refinedAnalysisBounds);
        var finalGeometry = CandidateScorer.MeasureRectangleGeometry(edges, refinedAnalysisBounds);
        var finalSharpness = CandidateScorer.MeasureSharpnessContrast(focusMap, refinedAnalysisBounds);
        var finalBoundaryContrast = CandidateScorer.MeasureBoundaryContrast(gray, refinedAnalysisBounds);
        var finalConfidence = ScoreFinalCandidate(
            finalBoundary, finalEdgeStrength, finalGeometry, finalSharpness, finalCornerFit,
            finalBoundaryContrast, bestMatch.FromFocusMap,
            refinedAnalysisBounds, analysis.Width, analysis.Height);

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

        var secondMatch = ranked.FirstOrDefault(candidate => !ReferenceEquals(candidate, bestMatch));
        var second = secondMatch?.Result;
        var margin = second is null ? 1.0 : bestMatch.Result.Confidence - second.Confidence;
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
        List<RankedCandidate> candidates,
        Mat edges,
        Mat gray,
        Mat focusMap,
        Rect bounds,
        double geometryHint,
        bool fromFocusMap,
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
        var confidence = ScoreCandidate(
            boundary, edgeStrength, geometry, sharpness, cornerFit, fromFocusMap,
            bounds, imageWidth, imageHeight);
        var detectionCandidate = new DetectionCandidate(
            DetectionGeometry.ScaleRect(bounds, 1.0 / scale), confidence, boundary, edgeStrength,
            geometry, sharpness, cornerFit, 0, areaRatio);
        var candidate = new RankedCandidate(detectionCandidate, fromFocusMap);

        var duplicateIndex = candidates.FindIndex(existing =>
            DetectionGeometry.IntersectionOverUnion(existing.Result.Bounds, detectionCandidate.Bounds) >= 0.82);
        if (duplicateIndex < 0)
            candidates.Add(candidate);
        else if (detectionCandidate.Confidence > candidates[duplicateIndex].Result.Confidence)
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
        double cornerFit,
        bool fromFocusMap,
        Rect bounds,
        int imageWidth,
        int imageHeight)
    {
        var centerPreference = CandidateScorer.MeasureCenterPreference(bounds, imageWidth, imageHeight);
        if (fromFocusMap)
            return boundary * 0.092 +
                   edgeStrength * 0.184 +
                   geometry * 0.092 +
                   sharpness * 0.506 +
                   cornerFit * 0.046 +
                   centerPreference * 0.08;

        return boundary * 0.276 +
               edgeStrength * 0.1104 +
               geometry * 0.184 +
               sharpness * 0.3036 +
               cornerFit * 0.046 +
               centerPreference * 0.08;
    }

    private static double ScoreFinalCandidate(
        double boundary,
        double edgeStrength,
        double geometry,
        double sharpness,
        double cornerFit,
        double boundaryContrast,
        bool fromFocusMap,
        Rect bounds,
        int imageWidth,
        int imageHeight)
    {
        var centerPreference = CandidateScorer.MeasureCenterPreference(bounds, imageWidth, imageHeight);
        if (fromFocusMap)
            return boundary * 0.22 +
                   edgeStrength * 0.10 +
                   geometry * 0.18 +
                   sharpness * 0.30 +
                   cornerFit * 0.04 +
                   centerPreference * 0.08 +
                   boundaryContrast * 0.08;

        return boundary * 0.30 +
               edgeStrength * 0.10 +
               geometry * 0.20 +
               sharpness * 0.24 +
               cornerFit * 0.04 +
               centerPreference * 0.06 +
               boundaryContrast * 0.06;
    }

    private static Task ReportProgress(int progress, Action<int>? reportProgress, Func<Task>? yieldToBrowser)
    {
        reportProgress?.Invoke(progress);
        return YieldToBrowser(yieldToBrowser);
    }

    private static Task YieldToBrowser(Func<Task>? yieldToBrowser) =>
        yieldToBrowser?.Invoke() ?? Task.CompletedTask;

    private sealed record RankedCandidate(DetectionCandidate Result, bool FromFocusMap);
}
