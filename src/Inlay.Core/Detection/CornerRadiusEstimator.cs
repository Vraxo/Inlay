using OpenCvSharp;

namespace Inlay.Detection;

internal static class CornerRadiusEstimator
{
    internal static int EstimateCornerRadius(Mat edges, Rect bounds)
    {
        var maximumRadius = (int)(Math.Min(bounds.Width, bounds.Height) * 0.15);
        if (maximumRadius < 8)
            return 0;

        var estimates = new List<int>(4);
        foreach (var (horizontalSign, verticalSign) in new[]
        {
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)
        })
        {
            var scores = new List<(int Radius, double Coverage)>(maximumRadius - 7);
            for (var radius = 8; radius <= maximumRadius; radius++)
            {
                var arc = MeasureCornerArcCoverage(edges, bounds, radius, horizontalSign, verticalSign);
                if (arc.Coverage >= 0.72 && arc.Continuity >= 0.80)
                    scores.Add((radius, arc.Coverage));
            }

            if (scores.Count == 0)
                continue;

            var strongestCoverage = scores.Max(score => score.Coverage);
            if (strongestCoverage < 0.72)
                continue;

            // A few adjacent radii can score equally because edges are rasterized.
            // The center of that plateau is a steadier estimate than its first pixel.
            var plateau = scores
                .Where(score => score.Coverage >= strongestCoverage - 0.03)
                .Select(score => score.Radius)
                .ToArray();
            estimates.Add((int)Math.Round(plateau.Average()));
        }

        // Rounded-corner evidence must be visible at every corner. Three
        // nearby text or image details are not enough to imply a rounded crop.
        if (estimates.Count != 4)
            return 0;

        estimates.Sort();
        var middle = estimates.Count / 2;
        var median = estimates.Count % 2 == 0
            ? (estimates[middle - 1] + estimates[middle]) / 2.0
            : estimates[middle];
        var tolerance = Math.Max(5, median * 0.20);
        var consistent = estimates.Where(radius => Math.Abs(radius - median) <= tolerance).ToArray();
        if (consistent.Length != 4)
            return 0;

        return (int)Math.Round(consistent.Average());
    }

    internal static (double Coverage, double Continuity) MeasureCornerArcCoverage(
        Mat edges,
        Rect bounds,
        int radius,
        int horizontalSign,
        int verticalSign)
    {
        var centerX = horizontalSign < 0
            ? bounds.Left + radius
            : bounds.Right - 1 - radius;
        var centerY = verticalSign < 0
            ? bounds.Top + radius
            : bounds.Bottom - 1 - radius;
        // Check most of the quarter-circle. A narrow central arc can be
        // accidentally matched by nearby text or graphics on square corners.
        var sampleCount = Math.Clamp(radius / 2, 12, 48);
        var firstAngle = Math.PI / 12;
        var lastAngle = Math.PI * 5 / 12;
        var hits = 0;
        var currentRun = 0;
        var longestRun = 0;

        for (var sample = 0; sample <= sampleCount; sample++)
        {
            var angle = firstAngle + (lastAngle - firstAngle) * sample / sampleCount;
            var x = (int)Math.Round(centerX + horizontalSign * radius * Math.Cos(angle));
            var y = (int)Math.Round(centerY + verticalSign * radius * Math.Sin(angle));
            if (HasNearbyEdge(edges, x, y))
            {
                hits++;
                currentRun++;
                longestRun = Math.Max(longestRun, currentRun);
            }
            else
            {
                currentRun = 0;
            }
        }

        var sampleTotal = sampleCount + 1;
        return (
            hits / (double)sampleTotal,
            longestRun / (double)sampleTotal
        );
    }

    internal static bool HasNearbyEdge(Mat edges, int x, int y)
    {
        for (var offsetY = -1; offsetY <= 1; offsetY++)
        {
            for (var offsetX = -1; offsetX <= 1; offsetX++)
            {
                var sampleX = x + offsetX;
                var sampleY = y + offsetY;
                if (sampleX >= 0 && sampleX < edges.Width &&
                    sampleY >= 0 && sampleY < edges.Height &&
                    edges.At<byte>(sampleY, sampleX) != 0)
                    return true;
            }
        }

        return false;
    }
}
