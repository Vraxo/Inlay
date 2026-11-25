using System.Buffers.Binary;
using System.Runtime.InteropServices.JavaScript;
using System.Runtime.Versioning;
using System.Text.Json;
using Inlay.Detection;
using Inlay.Processing;
using OpenCvSharp;

namespace Inlay.Web.Workers;

[SupportedOSPlatform("browser")]
public static partial class ImageProcessingWorker
{
    [JSImport("globalThis.postMessage")]
    private static partial void PostWorkerMessage(string message);

    [JSExport]
    public static byte[] ProcessImage(int requestId, byte[] sourceBytes)
    {
        using var source = Cv2.ImDecode(sourceBytes, ImreadModes.Color);
        if (source.Empty())
            throw new InvalidDataException("Could not decode this image.");

        var detection = new ImageDetector().DetectAsync(
            source,
            progress => PostProgress(requestId, progress))
            .GetAwaiter()
            .GetResult();

        var bounds = detection.Best?.Bounds ?? new Rect(0, 0, source.Width, source.Height);
        var radius = detection.Best?.CornerRadius ?? 0;
        var cropBytes = ImageProcessing.CreateCrop(
            source, bounds.X, bounds.Y, bounds.Width, bounds.Height, radius);
        var thumbnailBytes = ImageProcessing.CreateThumbnail(cropBytes);

        var metadata = JsonSerializer.SerializeToUtf8Bytes(new WorkerMetadata(
            source.Width,
            source.Height,
            detection.Status.ToString(),
            detection.Reason,
            CreateCandidate(detection.Best),
            CreateCandidate(detection.Second)));

        var output = new byte[12 + metadata.Length + cropBytes.Length + thumbnailBytes.Length];
        BinaryPrimitives.WriteInt32LittleEndian(output.AsSpan(0, 4), metadata.Length);
        BinaryPrimitives.WriteInt32LittleEndian(output.AsSpan(4, 4), cropBytes.Length);
        BinaryPrimitives.WriteInt32LittleEndian(output.AsSpan(8, 4), thumbnailBytes.Length);
        metadata.CopyTo(output, 12);
        cropBytes.CopyTo(output, 12 + metadata.Length);
        thumbnailBytes.CopyTo(output, 12 + metadata.Length + cropBytes.Length);
        return output;
    }

    private static void PostProgress(int requestId, int progress)
    {
        PostWorkerMessage(JsonSerializer.Serialize(new { type = "progress", requestId, progress }));
    }

    private static WorkerCandidate? CreateCandidate(DetectionCandidate? candidate)
    {
        if (candidate is null)
            return null;

        var bounds = candidate.Bounds;
        return new WorkerCandidate(
            bounds.X, bounds.Y, bounds.Width, bounds.Height,
            candidate.Confidence, candidate.BoundaryContinuity, candidate.EdgeStrength,
            candidate.GeometryFit, candidate.SharpnessContrast, candidate.CornerFit,
            candidate.CornerRadius, candidate.AreaRatio);
    }

    private sealed record WorkerMetadata(
        int SourceWidth, int SourceHeight, string Status, string Reason,
        WorkerCandidate? Best, WorkerCandidate? Second);

    private sealed record WorkerCandidate(
        int X, int Y, int Width, int Height, double Confidence,
        double BoundaryContinuity, double EdgeStrength, double GeometryFit,
        double SharpnessContrast, double CornerFit, int CornerRadius, double AreaRatio);
}
