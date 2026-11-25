using OpenCvSharp;
using Inlay.Extraction;

namespace Inlay.Processing;

public static class ImageProcessing
{
    public static byte[] CreateCrop(byte[] sourceBytes, int x, int y, int width, int height, int radius)
    {
        using var source = Cv2.ImDecode(sourceBytes, ImreadModes.Color);
        if (source.Empty())
            throw new InvalidDataException("Could not decode the source image.");

        return CreateCrop(source, x, y, width, height, radius);
    }

    public static byte[] CreateCrop(Mat source, int x, int y, int width, int height, int radius)
    {
        if (source.Empty() || x < 0 || y < 0 || width <= 0 || height <= 0 ||
            (long)x + width > source.Width || (long)y + height > source.Height)
            throw new ArgumentOutOfRangeException(nameof(width), "The crop rectangle must fit inside the source image.");

        if (radius < 0 || radius > Math.Min(width, height) / 2)
            throw new ArgumentOutOfRangeException(nameof(radius));

        using var crop = new Mat(source, new Rect(x, y, width, height));
        using var bgra = new Mat();
        Cv2.CvtColor(crop, bgra, ColorConversionCodes.BGR2BGRA);
        using var mask = ImageExtractor.CreateRoundedMask(width, height, radius);
        Cv2.InsertChannel(mask, bgra, 3);

        if (!Cv2.ImEncode(".png", bgra, out var encodedCrop))
            throw new InvalidOperationException("Could not encode the crop as PNG.");

        return encodedCrop;
    }

    public static byte[] CreateThumbnail(byte[] cropBytes)
    {
        using var source = Cv2.ImDecode(cropBytes, ImreadModes.Unchanged);
        if (source.Empty())
            throw new InvalidDataException("Could not decode the crop thumbnail.");

        const int maximumDimension = 144;
        var scale = Math.Min(1d, maximumDimension / (double)Math.Max(source.Width, source.Height));
        using var thumbnail = new Mat();

        if (scale < 1)
        {
            var size = new Size(
                Math.Max(1, (int)Math.Round(source.Width * scale)),
                Math.Max(1, (int)Math.Round(source.Height * scale)));
            Cv2.Resize(source, thumbnail, size, interpolation: InterpolationFlags.Area);
        }
        else
        {
            source.CopyTo(thumbnail);
        }

        if (!Cv2.ImEncode(".png", thumbnail, out var encodedThumbnail))
            throw new InvalidOperationException("Could not encode the crop thumbnail.");

        return encodedThumbnail;
    }
}
