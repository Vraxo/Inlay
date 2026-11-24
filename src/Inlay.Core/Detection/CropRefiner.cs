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

    internal static Rect RefineTopBoundary(Mat edges, Rect bounds, int searchRadius)
    {
        var horizontalInset = Math.Max(2, bounds.Width / 10);
        var lineX = bounds.X + horizontalInset;
        var lineWidth = bounds.Width - horizontalInset * 2;
        if (lineWidth < 20 || bounds.Height < 20)
            return bounds;

        // Canny locates the transition on the final background row. The crop
        // begins on the following row, so convert edge rows to pixel bounds.
        var firstEdgeRow = Math.Max(0, bounds.Y - searchRadius - 1);
        var lastEdgeRow = Math.Min(edges.Height - 1, bounds.Y + searchRadius - 1);
        var bestEdgeRow = bounds.Y - 1;
        var bestStrength = bestEdgeRow >= 0
            ? DetectionGeometry.CountEdges(edges, new Rect(lineX, bestEdgeRow, lineWidth, 1)) / (double)lineWidth
            : 0;

        for (var row = firstEdgeRow; row <= lastEdgeRow; row++)
        {
            var strength = DetectionGeometry.CountEdges(edges, new Rect(lineX, row, lineWidth, 1)) /
                           (double)lineWidth;
            if (strength > bestStrength)
            {
                bestEdgeRow = row;
                bestStrength = strength;
            }
        }

        // If the strongest edge lies below the original candidate top,
        // it marks the first foreground row. If it lies at/above that top,
        // it marks the final background row and the crop starts one row later.
        var refinedY = bestEdgeRow > bounds.Y ? bestEdgeRow : bestEdgeRow + 1;
        var refinedHeight = bounds.Bottom - refinedY;
        return bestStrength >= 0.05 && refinedY != bounds.Y && refinedHeight > 0
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

        // The reported crop extends too far downward. Search only above its
        // current bottom so this pass can reduce height, never increase it.
        var originalBottomRow = bounds.Bottom - 1;
        var firstEdgeRow = Math.Max(0, originalBottomRow - searchRadius);
        var bestEdgeRow = originalBottomRow;
        var originalStrength = DetectionGeometry.CountEdges(edges, new Rect(lineX, originalBottomRow, lineWidth, 1)) /
                               (double)lineWidth;
        var bestStrength = originalStrength;

        for (var row = firstEdgeRow; row < originalBottomRow; row++)
        {
            var strength = DetectionGeometry.CountEdges(edges, new Rect(lineX, row, lineWidth, 1)) /
                           (double)lineWidth;
            if (strength > bestStrength)
            {
                bestEdgeRow = row;
                bestStrength = strength;
            }
        }

        var meaningfulImprovement = bestEdgeRow < originalBottomRow &&
                                    bestStrength - originalStrength >= 0.04 &&
                                    bestStrength >= originalStrength * 1.25 &&
                                    bestStrength >= 0.05;
        if (!meaningfulImprovement)
            return bounds;

        // The selected edge row is the final foreground row, so the bottom
        // coordinate is exclusive and sits one pixel after that edge.
        var bottomExclusive = bestEdgeRow + 1;
        if (bottomExclusive <= bounds.Y || bottomExclusive >= bounds.Bottom)
            return bounds;

        return new Rect(bounds.X, bounds.Y, bounds.Width, bottomExclusive - bounds.Y);
    }
}
