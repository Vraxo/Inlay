using Inlay.Processing;
using OpenCvSharp;
using Xunit;

namespace Inlay.Tests;

public sealed class ImageProcessingTests
{
    [Fact]
    public void CropPreservesRequestedPixelBoundsAndTransparency()
    {
        using var source = new Mat(420, 640, MatType.CV_8UC3, new Scalar(30, 60, 90));
        var bytes = ImageProcessing.CreateCrop(source, 80, 90, 240, 180, 24);
        using var crop = Cv2.ImDecode(bytes, ImreadModes.Unchanged);

        Assert.Equal(240, crop.Width);
        Assert.Equal(180, crop.Height);
        Assert.Equal(MatType.CV_8UC4, crop.Type());
        Assert.Equal(0, crop.At<Vec4b>(0, 0).Item3);
        Assert.Equal(255, crop.At<Vec4b>(90, 120).Item3);
    }

    [Fact]
    public void ThumbnailDoesNotExceedItsMaximumDimension()
    {
        using var source = new Mat(800, 1200, MatType.CV_8UC3, new Scalar(40, 80, 120));
        var crop = ImageProcessing.CreateCrop(source, 0, 0, source.Width, source.Height, 0);
        var thumbnailBytes = ImageProcessing.CreateThumbnail(crop);
        using var thumbnail = Cv2.ImDecode(thumbnailBytes, ImreadModes.Unchanged);

        Assert.Equal(144, thumbnail.Width);
        Assert.Equal(96, thumbnail.Height);
    }
}
