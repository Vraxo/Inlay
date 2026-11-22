using Inlay.Extraction;
using OpenCvSharp;
using Xunit;

namespace Inlay.Tests;

public sealed class ImageExtractorTests
{
    [Fact]
    public void ManualCropUsesExactPixelBounds()
    {
        var path = Path.Combine(Path.GetTempPath(), $"inlay-{Guid.NewGuid():N}.png");
        using var source = new Mat(40, 60, MatType.CV_8UC3, new Scalar(12, 34, 56));

        try
        {
            new ImageExtractor().Save(source, new Rect(7, 9, 20, 15), path, 0, false);
            using var result = Cv2.ImRead(path, ImreadModes.Unchanged);

            Assert.Equal(20, result.Width);
            Assert.Equal(15, result.Height);
            Assert.Equal(60, source.Width);
            Assert.Equal(40, source.Height);
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }
}
