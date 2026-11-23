using System.IO.Compression;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Forms;
using Microsoft.JSInterop;
using OpenCvSharp;
using Inlay.Detection;
using Inlay.Extraction;

namespace Inlay.Web.Pages;

public partial class Home
{
    private const long MaximumFileSize = 25 * 1024 * 1024;
    private const long MaximumBatchSize = 50 * 1024 * 1024;
    private const int MaximumFiles = 20;

    private readonly ImageDetector detector = new();
    private readonly List<BatchImage> images = [];
    private BatchImage? activeImage;
    private string error = "";
    private string validationMessage = "";
    private bool isProcessing;
    private int sourceWidth;
    private int sourceHeight;
    private int cropX;
    private int cropY;
    private int cropWidth;
    private int cropHeight;
    private int cornerRadius;

    private string StatusClass => GetStatusClass(activeImage?.Detection);
    private int SuccessfulCount => images.Count(image => image.CropBytes is not null);
    private int ReviewCount => images.Count(image => image.Error.Length == 0 && (image.Detection?.Status != DetectionStatus.Extracted || image.Detection?.Best is null));
    private int FailedCount => images.Count(image => image.Error.Length > 0);
    private bool HasResults => images.Count > 0;

    private static string GetStatusClass(DetectionResult? result) => result?.Status switch
    {
        DetectionStatus.Extracted => "success",
        DetectionStatus.Review => "review",
        _ => "rejected"
    };

    private static string GetStatusLabel(BatchImage image)
    {
        if (!string.IsNullOrEmpty(image.Error))
            return "Failed";

        return image.Detection?.Status switch
        {
            DetectionStatus.Extracted => "Extracted",
            DetectionStatus.Review => "Review",
            _ => "Manual crop"
        };
    }

    private async Task LoadImages(InputFileChangeEventArgs args)
    {
        images.Clear();
        activeImage = null;
        error = "";
        validationMessage = "";

        IReadOnlyList<IBrowserFile> files;
        try
        {
            files = args.GetMultipleFiles(MaximumFiles);
        }
        catch (InvalidOperationException)
        {
            error = $"Choose no more than {MaximumFiles} images at a time.";
            return;
        }

        if (files.Sum(file => file.Size) > MaximumBatchSize)
        {
            error = "The selected files exceed the 50 MB total batch limit. Choose fewer or smaller images.";
            return;
        }

        isProcessing = true;
        try
        {
            foreach (var file in files)
            {
                var item = new BatchImage { FileName = file.Name };
                images.Add(item);

                if (file.Size > MaximumFileSize)
                {
                    item.Error = "Larger than 25 MB.";
                    continue;
                }

                if (file.ContentType is not ("image/png" or "image/jpeg" or "image/webp"))
                {
                    item.Error = "Unsupported format.";
                    continue;
                }

                try
                {
                    await using var stream = file.OpenReadStream(MaximumFileSize);
                    using var memory = new MemoryStream();
                    await stream.CopyToAsync(memory);
                    item.SourceBytes = memory.ToArray();
                    item.SourcePreview = $"data:{file.ContentType};base64,{Convert.ToBase64String(item.SourceBytes)}";

                    using var source = Cv2.ImDecode(item.SourceBytes, ImreadModes.Color);
                    if (source.Empty())
                        throw new InvalidDataException("Could not decode this image.");

                    item.SourceWidth = source.Width;
                    item.SourceHeight = source.Height;
                    item.Detection = detector.Detect(source);

                    if (item.Detection.Best is not null)
                    {
                        item.CropX = item.Detection.Best.Bounds.X;
                        item.CropY = item.Detection.Best.Bounds.Y;
                        item.CropWidth = item.Detection.Best.Bounds.Width;
                        item.CropHeight = item.Detection.Best.Bounds.Height;
                        item.CornerRadius = item.Detection.Best.CornerRadius;
                    }
                    else
                    {
                        item.CropWidth = source.Width;
                        item.CropHeight = source.Height;
                    }

                    item.CropBytes = CreateCrop(item, item.CropX, item.CropY, item.CropWidth, item.CropHeight, item.CornerRadius);
                    item.CropPreview = $"data:image/png;base64,{Convert.ToBase64String(item.CropBytes)}";
                }
                catch (Exception exception)
                {
                    item.Error = $"Could not process: {exception.Message}";
                }
            }

            var firstUsable = images.FirstOrDefault(image => image.SourceBytes is not null);
            if (firstUsable is not null)
                SelectImage(firstUsable);
        }
        finally
        {
            isProcessing = false;
        }
    }

    private void SelectImage(BatchImage image)
    {
        SaveActiveImage();
        activeImage = image;
        sourceWidth = image.SourceWidth;
        sourceHeight = image.SourceHeight;
        cropX = image.CropX;
        cropY = image.CropY;
        cropWidth = image.CropWidth;
        cropHeight = image.CropHeight;
        cornerRadius = image.CornerRadius;
        validationMessage = "";
    }

    private void SaveActiveImage()
    {
        if (activeImage is null)
            return;

        activeImage.CropX = cropX;
        activeImage.CropY = cropY;
        activeImage.CropWidth = cropWidth;
        activeImage.CropHeight = cropHeight;
        activeImage.CornerRadius = cornerRadius;
    }

    private void UseDetectionBounds()
    {
        if (activeImage?.Detection?.Best is null)
            return;

        cropX = activeImage.Detection.Best.Bounds.X;
        cropY = activeImage.Detection.Best.Bounds.Y;
        cropWidth = activeImage.Detection.Best.Bounds.Width;
        cropHeight = activeImage.Detection.Best.Bounds.Height;
        cornerRadius = activeImage.Detection.Best.CornerRadius;
        ApplyCrop();
    }

    private void ApplyCrop()
    {
        validationMessage = "";
        if (activeImage?.SourceBytes is null)
            return;

        activeImage.CropBytes = null;
        activeImage.CropPreview = "";

        if (cropX < 0 || cropY < 0 || cropWidth <= 0 || cropHeight <= 0 ||
            cropX >= sourceWidth || cropY >= sourceHeight ||
            (long)cropX + cropWidth > sourceWidth || (long)cropY + cropHeight > sourceHeight)
        {
            validationMessage = "The crop rectangle must fit inside the original image.";
            return;
        }

        if (cornerRadius < 0 || cornerRadius > Math.Min(cropWidth, cropHeight) / 2)
        {
            validationMessage = "The corner radius is outside the allowed range.";
            return;
        }

        try
        {
            activeImage.CropBytes = CreateCrop(activeImage, cropX, cropY, cropWidth, cropHeight, cornerRadius);
            activeImage.CropPreview = $"data:image/png;base64,{Convert.ToBase64String(activeImage.CropBytes)}";
            SaveActiveImage();
        }
        catch (Exception exception)
        {
            activeImage.CropBytes = null;
            activeImage.CropPreview = "";
            validationMessage = $"Could not create crop: {exception.Message}";
        }
    }

    private static byte[] CreateCrop(BatchImage image, int x, int y, int width, int height, int radius)
    {
        using var source = Cv2.ImDecode(image.SourceBytes!, ImreadModes.Color);
        using var crop = new Mat(source, new Rect(x, y, width, height));
        using var bgra = new Mat();
        Cv2.CvtColor(crop, bgra, ColorConversionCodes.BGR2BGRA);
        using var mask = ImageExtractor.CreateRoundedMask(width, height, radius);
        Cv2.InsertChannel(mask, bgra, 3);

        if (!Cv2.ImEncode(".png", bgra, out var encodedCrop))
            throw new InvalidOperationException("Could not encode the crop as PNG.");

        return encodedCrop;
    }

    private async Task DownloadCrop()
    {
        if (activeImage?.CropBytes is null)
            return;

        await Download(activeImage.CropBytes, GetOutputName(activeImage.FileName), "image/png");
    }

    private async Task DownloadAll()
    {
        SaveActiveImage();
        var ready = images.Where(image => image.CropBytes is not null).ToList();
        if (ready.Count == 0)
            return;

        using var archiveStream = new MemoryStream();
        using (var archive = new ZipArchive(archiveStream, ZipArchiveMode.Create, leaveOpen: true))
        {
            var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var image in ready)
            {
                var baseName = GetOutputName(image.FileName);
                var outputName = baseName;
                var suffix = 2;
                while (!names.Add(outputName))
                    outputName = $"{Path.GetFileNameWithoutExtension(baseName)}-{suffix++}.png";

                var entry = archive.CreateEntry(outputName, CompressionLevel.Fastest);
                await using var entryStream = entry.Open();
                await entryStream.WriteAsync(image.CropBytes!);
            }
        }

        await Download(archiveStream.ToArray(), "inlay-extractions.zip", "application/zip");
    }

    private async Task Download(byte[] bytes, string fileName, string contentType)
    {
        await using var stream = new MemoryStream(bytes);
        using var streamReference = new DotNetStreamReference(stream);
        await JS.InvokeVoidAsync("inlay.download", fileName, streamReference, contentType);
    }

    private static string GetOutputName(string fileName) =>
        $"{Path.GetFileNameWithoutExtension(fileName)}.inlay.png";

    private sealed class BatchImage
    {
        public string FileName { get; init; } = "";
        public string Error { get; set; } = "";
        public byte[]? SourceBytes { get; set; }
        public byte[]? CropBytes { get; set; }
        public string SourcePreview { get; set; } = "";
        public string CropPreview { get; set; } = "";
        public int SourceWidth { get; set; }
        public int SourceHeight { get; set; }
        public int CropX { get; set; }
        public int CropY { get; set; }
        public int CropWidth { get; set; }
        public int CropHeight { get; set; }
        public int CornerRadius { get; set; }
        public DetectionResult? Detection { get; set; }
    }
}
