using Inlay.Detection;

namespace Inlay.Web.Pages;

public enum BatchStatus
{
    Queued, Reading, Decoding, Detecting, CreatingPreview, Completed, Failed
}

public sealed class BatchImage
{
    private string error = "";
    private string thumbnailPreview = "";
    private int progress;
    private BatchStatus status = BatchStatus.Queued;

    public string Id { get; } = Guid.NewGuid().ToString("N");
    public string FileName { get; init; } = "";
    public byte[]? SourceBytes { get; set; }
    public byte[]? CropBytes { get; set; }
    public string SourceContentType { get; set; } = "image/png";
    public string SourcePreview { get; set; } = "";
    public string ThumbnailPreview
    {
        get => thumbnailPreview;
        set
        {
            if (thumbnailPreview == value) return;
            thumbnailPreview = value;
            Changed?.Invoke(this, EventArgs.Empty);
        }
    }
    public int SourceWidth { get; set; }
    public int SourceHeight { get; set; }
    public int CropX { get; set; }
    public int CropY { get; set; }
    public int CropWidth { get; set; }
    public int CropHeight { get; set; }
    public int CornerRadius { get; set; }
    public int Progress
    {
        get => progress;
        set
        {
            var next = Math.Clamp(value, 0, 100);
            if (progress == next) return;
            progress = next;
            Changed?.Invoke(this, EventArgs.Empty);
        }
    }
    public BatchStatus Status
    {
        get => status;
        set
        {
            if (status == value) return;
            status = value;
            Changed?.Invoke(this, EventArgs.Empty);
        }
    }
    public string Error
    {
        get => error;
        set
        {
            if (error == value) return;
            error = value;
            Changed?.Invoke(this, EventArgs.Empty);
        }
    }
    public bool IsComplete => Status is BatchStatus.Completed or BatchStatus.Failed;
    public bool CanSelect => Status == BatchStatus.Completed &&
        SourceBytes is not null && SourceWidth > 0 && SourceHeight > 0;
    public DetectionResult? Detection { get; set; }

    public event EventHandler? Changed;
}
