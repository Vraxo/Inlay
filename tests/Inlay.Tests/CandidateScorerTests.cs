using Inlay.Detection;
using OpenCvSharp;
using Xunit;

namespace Inlay.Tests;

public sealed class CandidateScorerTests
{
    [Fact]
    public void BoundaryMetricsTolerateSmallAlignmentErrors()
    {
        using var edges = new Mat(700, 700, MatType.CV_8UC1, Scalar.Black);
        var suggested = new Rect(100, 100, 500, 500);
        var actual = new Rect(97, 103, 506, 494);
        Cv2.Rectangle(edges, actual, Scalar.White, 2);

        var continuity = CandidateScorer.MeasureBoundaryContinuity(edges, suggested);
        var geometry = CandidateScorer.MeasureRectangleGeometry(edges, suggested);

        Assert.True(continuity >= 0.80, $"Expected tolerant boundary continuity, got {continuity:F3}.");
        Assert.True(geometry >= 0.80, $"Expected tolerant rectangle geometry, got {geometry:F3}.");
    }

    [Fact]
    public void BoundaryContrastFindsAForegroundEdgeA few pixelsAway()
    {
        using var gray = new Mat(700, 700, MatType.CV_8UC1, new Scalar(25));
        var suggested = new Rect(100, 100, 500, 500);
        var actual = new Rect(97, 103, 506, 494);
        using (var foreground = new Mat(gray, actual))
            foreground.SetTo(new Scalar(190));

        var contrast = CandidateScorer.MeasureBoundaryContrast(gray, suggested);

        Assert.True(contrast >= 0.90, $"Expected strong four-sided boundary contrast, got {contrast:F3}.");
    }

    [Fact]
    public void BoundaryContrastDoesNotInventEdgesInAUniformImage()
    {
        using var gray = new Mat(700, 700, MatType.CV_8UC1, new Scalar(90));
        var suggested = new Rect(100, 100, 500, 500);

        var contrast = CandidateScorer.MeasureBoundaryContrast(gray, suggested);

        Assert.Equal(0, contrast);
    }
}
