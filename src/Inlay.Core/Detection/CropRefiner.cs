using OpenCvSharp;

namespace Inlay.Detection;

internal static class CropRefiner
{
    internal static Rect RefineHorizontalBounds(Mat edges, Rect bounds)
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
        var originalBoundary = CandidateScorer.MeasureBoundaryContinuity(edges, bounds);
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

                var top = DetectionGeometry.CountEdges(edges, new Rect(horizontalX, bounds.Y, horizontalWidth, thickness)) /
                          (horizontalWidth * (double)thickness);
                var bottom = DetectionGeometry.CountEdges(edges, new Rect(horizontalX, bounds.Bottom - thickness, horizontalWidth, thickness)) /
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

    internal static Rect RefineTopBoundary(Mat edges, Mat focusMap, Rect bounds, int searchRadius)
    {
        var horizontalInset = Math.Max(2, bounds.Width / 10);
        var lineX = bounds.X + horizontalInset;
        var lineWidth = bounds.Width - horizontalInset * 2;
        if (lineWidth < 20 || bounds.Height < 20 ||
            focusMap.Width != edges.Width || focusMap.Height != edges.Height)
            return bounds;

        var bandHeight = Math.Clamp(lineWidth / 40, 12, 32);
        var firstEdgeRow = Math.Max(bandHeight, bounds.Y - searchRadius - 1);
        var lastEdgeRow = Math.Min(edges.Height - bandHeight - 2, bounds.Y + searchRadius - 1);
        if (firstEdgeRow > lastEdgeRow)
            return bounds;

        using var focusRegion = new Mat(focusMap, new Rect(lineX, 0, lineWidth, focusMap.Height));
        using var rowAverages = new Mat();
        Cv2.Reduce(focusRegion, rowAverages, 1, ReduceTypes.Avg, MatType.CV_32F);

        var prefix = new double[focusMap.Height + 1];
        for (var row = 0; row < focusMap.Height; row++)
            prefix[row + 1] = prefix[row] + rowAverages.At<float>(row, 0);

        var bestEdgeRow = bounds.Y - 1;
        var bestScore = 0.0;
        var bestTransition = 0.0;
        var bestDifference = 0.0;

        for (var row = firstEdgeRow; row <= lastEdgeRow; row++)
        {
            var outside = (prefix[row] - prefix[row - bandHeight]) / bandHeight;
            var inside = (prefix[row + bandHeight + 1] - prefix[row + 1]) / bandHeight;
            var difference = inside - outside;
            var transition = difference / Math.Max(8.0, inside + outside);
            var boundaryStrength = DetectionGeometry.CountEdges(edges, new Rect(lineX, row, lineWidth, 1)) /
                                   (double)lineWidth;
            var score = transition * 0.85 + Math.Min(boundaryStrength / 0.10, 1.0) * 0.15;

            if (score <= bestScore)
                continue;

            bestEdgeRow = row;
            bestScore = score;
            bestTransition = transition;
            bestDifference = difference;
        }

        // The focus map measures high-frequency detail across the whole image
        // width. A photo boundary creates a sustained jump from its blurred
        // backdrop, even when the exact border has little Canny edge response.
        // This lets us search beyond the old 24 px local-edge fallback without
        // mistaking a single strong line inside the photograph for its edge.
        if (bestTransition < 0.12 || bestDifference < 2.0)
            return RefineTopBoundaryNearOriginal(edges, bounds, searchRadius, lineX, lineWidth);

        var refinedY = bestEdgeRow + 1;
        var refinedHeight = bounds.Bottom - refinedY;
        return refinedY != bounds.Y && refinedHeight > 0
            ? new Rect(bounds.X, refinedY, bounds.Width, refinedHeight)
            : bounds;
    }

    internal static Rect RefineBottomBoundary(Mat edges, Rect bounds, int searchRadius)
    {
        var horizontalInset = Math.Max(2, bounds.Width / 10);
        var lineX = bounds.X + horizontalInset;
        var lineWidth = bounds.Width - horizontalInset * 2;
        if (lineWidth < 20 || bounds.Height < 20)
            return bounds;

        var bandHeight = Math.Clamp(lineWidth / 40, 12, 32);
        var originalBottomRow = bounds.Bottom - 1;
        var firstEdgeRow = Math.Max(bandHeight, originalBottomRow - searchRadius);
        var lastEdgeRow = Math.Min(originalBottomRow, edges.Height - bandHeight - 1);
        var bestEdgeRow = originalBottomRow;
        var bestScore = 0.0;
        var bestTransition = 0.0;
        var bestInsideDensity = 0.0;
        var bestBoundaryStrength = 0.0;

        for (var row = firstEdgeRow; row <= lastEdgeRow; row++)
        {
            var insideDensity = MeasureEdgeDensity(edges, lineX, row - bandHeight + 1, lineWidth, bandHeight);
            var outsideDensity = MeasureEdgeDensity(edges, lineX, row + 1, lineWidth, bandHeight);
            var transition = (insideDensity - outsideDensity) /
                             Math.Max(0.02, insideDensity + outsideDensity);
            var boundaryStrength = DetectionGeometry.CountEdges(edges, new Rect(lineX, row, lineWidth, 1)) /
                                   (double)lineWidth;
            var score = transition * 0.8 + Math.Min(boundaryStrength / 0.12, 1.0) * 0.2;

            if (score <= bestScore)
                continue;

            bestEdgeRow = row;
            bestScore = score;
            bestTransition = transition;
            bestInsideDensity = insideDensity;
            bestBoundaryStrength = boundaryStrength;
        }

        if (bestEdgeRow == originalBottomRow || bestTransition < 0.12 ||
            bestInsideDensity < 0.02 || bestBoundaryStrength < 0.015)
            return RefineBottomBoundaryNearOriginal(edges, bounds, searchRadius, lineX, lineWidth);

        var bottomExclusive = bestEdgeRow + 1;
        if (bottomExclusive <= bounds.Y || bottomExclusive >= bounds.Bottom)
            return bounds;

        return new Rect(bounds.X, bounds.Y, bounds.Width, bottomExclusive - bounds.Y);
    }

    private static Rect RefineTopBoundaryNearOriginal(
        Mat edges, Rect bounds, int searchRadius, int lineX, int lineWidth)
    {
        var radius = Math.Min(searchRadius, 24);
        var firstRow = Math.Max(0, bounds.Y - radius - 1);
        var lastRow = Math.Min(edges.Height - 1, bounds.Y + radius - 1);
        var bestRow = bounds.Y - 1;
        var bestStrength = bestRow >= 0
            ? DetectionGeometry.CountEdges(edges, new Rect(lineX, bestRow, lineWidth, 1)) / (double)lineWidth
            : 0;

        for (var row = firstRow; row <= lastRow; row++)
        {
            var strength = DetectionGeometry.CountEdges(edges, new Rect(lineX, row, lineWidth, 1)) /
                           (double)lineWidth;
            if (strength > bestStrength)
            {
                bestRow = row;
                bestStrength = strength;
            }
        }

        var refinedY = bestRow > bounds.Y ? bestRow : bestRow + 1;
        var refinedHeight = bounds.Bottom - refinedY;
        return bestStrength >= 0.05 && refinedY != bounds.Y && refinedHeight > 0
            ? new Rect(bounds.X, refinedY, bounds.Width, refinedHeight)
            : bounds;
    }

    private static Rect RefineBottomBoundaryNearOriginal(
        Mat edges, Rect bounds, int searchRadius, int lineX, int lineWidth)
    {
        var originalBottomRow = bounds.Bottom - 1;
        var firstRow = Math.Max(0, originalBottomRow - Math.Min(searchRadius, 24));
        var bestRow = originalBottomRow;
        var originalStrength = DetectionGeometry.CountEdges(edges, new Rect(lineX, originalBottomRow, lineWidth, 1)) /
                               (double)lineWidth;
        var bestStrength = originalStrength;

        for (var row = firstRow; row < originalBottomRow; row++)
        {
            var strength = DetectionGeometry.CountEdges(edges, new Rect(lineX, row, lineWidth, 1)) /
                           (double)lineWidth;
            if (strength > bestStrength)
            {
                bestRow = row;
                bestStrength = strength;
            }
        }

        var meaningfulImprovement = bestRow < originalBottomRow &&
                                    bestStrength - originalStrength >= 0.04 &&
                                    bestStrength >= originalStrength * 1.25 &&
                                    bestStrength >= 0.05;
        if (!meaningfulImprovement)
            return bounds;

        var bottomExclusive = bestRow + 1;
        return bottomExclusive > bounds.Y && bottomExclusive < bounds.Bottom
            ? new Rect(bounds.X, bounds.Y, bounds.Width, bottomExclusive - bounds.Y)
            : bounds;
    }

    private static double MeasureEdgeDensity(Mat edges, int x, int y, int width, int height)
    {
        var count = 0;
        for (var row = y; row < y + height; row++)
        {
            for (var column = x; column < x + width; column++)
            {
                if (edges.At<byte>(row, column) != 0)
                    count++;
            }
        }

        return count / (double)(width * height);
    }
}
