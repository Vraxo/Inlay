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

            var boundary = CandidateScorer.MeasureBoundaryContinuity(edges, bounds);
            var edgeStrength = CandidateScorer.MeasureEdgeStrength(gray, bounds);
            var sharpness = CandidateScorer.MeasureSharpnessContrast(gray, bounds);
            const double cornerFit = 0.65;

            var areaFit = Math.Clamp(1.0 - Math.Abs(areaRatio - 0.35) / 0.65, 0.25, 1.0);
            var confidence = boundary * 0.34 + edgeStrength * 0.18 + geometry * 0.18 +
                             sharpness * 0.18 + cornerFit * 0.07 + areaFit * 0.05;

            var candidate = new DetectionCandidate(
                DetectionGeometry.ScaleRect(bounds, 1.0 / scale), confidence, boundary, edgeStrength,
                geometry, sharpness, cornerFit, 0, areaRatio);

            if (candidates.All(existing => DetectionGeometry.IntersectionOverUnion(existing.Bounds, candidate.Bounds) < 0.82))
                candidates.Add(candidate);
            else
            {
                var duplicateIndex = candidates.FindIndex(existing => DetectionGeometry.IntersectionOverUnion(existing.Bounds, candidate.Bounds) >= 0.82);
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
        var analysisBounds = DetectionGeometry.ScaleRect(best.Bounds, scale);
        var refinedAnalysisBounds = CropRefiner.RefineHorizontalBounds(edges, analysisBounds);
        var refinedBounds = DetectionGeometry.ScaleRect(refinedAnalysisBounds, 1.0 / scale);
        var finalBounds = new Rect(refinedBounds.X, best.Bounds.Y, refinedBounds.Width, best.Bounds.Height);

        // Recheck only the top edge at source resolution. The analysis image
        // is downscaled, so a one-pixel correction can be lost during scaling.
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

        // Confidence must describe the rectangle shown to the user, not the
        // original contour that was used to discover its vertical position.
        var finalBoundary = CandidateScorer.MeasureBoundaryContinuity(edges, refinedAnalysisBounds);
        var finalEdgeStrength = CandidateScorer.MeasureEdgeStrength(gray, refinedAnalysisBounds);
        var finalGeometry = CandidateScorer.MeasureRectangleGeometry(edges, refinedAnalysisBounds);
        var finalSharpness = CandidateScorer.MeasureSharpnessContrast(gray, refinedAnalysisBounds);
        var finalAreaRatio = (double)(refinedAnalysisBounds.Width * refinedAnalysisBounds.Height) /
                        (analysis.Width * analysis.Height);
        var finalAreaFit = Math.Clamp(1.0 - Math.Abs(finalAreaRatio - 0.35) / 0.65, 0.25, 1.0);
        // The final score emphasizes the crop's four actual borders. Interior
        // texture and sharpness are supporting evidence, not proof of alignment.
        var finalConfidence = finalBoundary * 0.42 + finalEdgeStrength * 0.08 + finalGeometry * 0.20 +
                              finalSharpness * 0.13 + finalCornerFit * 0.07 + finalAreaFit * 0.10;
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
            AreaRatio = finalAreaRatio
        };

        var second = ranked.Length > 1 ? ranked[1] : null;
        var margin = second is null ? 1.0 : best.Confidence - second.Confidence;

        if (best.Confidence < DetectionResult.AutomaticThreshold || margin < 0.08)
            return new DetectionResult(DetectionStatus.Review, best, second,
                "A crop candidate was found. Check the suggested rectangle before downloading.");

        return new DetectionResult(DetectionStatus.Extracted, best, second,
            "Candidate passed the initial heuristic thresholds.");
    }

























        (int)Math.Round(rect.X * scale), (int)Math.Round(rect.Y * scale),
        (int)Math.Round(rect.Width * scale), (int)Math.Round(rect.Height * scale));




}
