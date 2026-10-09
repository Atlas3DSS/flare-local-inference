# Native JPEG and EXIF HTTP checks

The Rust frontend admits all 26 encoded-image fixtures from
[`qwen38_jpeg_exif.json`](../../tests/frontend/qwen38_jpeg_exif.json) through
the actual Qwen HTTP service. Four malformed or unsupported image URLs return
HTTP 400. Coverage includes baseline/progressive RGB JPEG with 4:4:4, 4:2:2,
and 4:2:0 sampling, grayscale and CMYK JPEG, and JPEG/PNG images with all eight
EXIF orientations. JSON and SSE requests alternate.

CPU fixture checks produce exactly the Pillow 12.3 RGB bytes for all 26 cases
and pass actual-checkpoint image preprocessing/admission. The native decoder
uses libjpeg-turbo with accurate IDCT and smooth chroma upsampling. The HTTP
service uses the pinned Qwen checkpoint, the static Qwen model contract, copy
speculation, and the 4,096-token prefill catalog.

The requests use temperature 1, top-k 20, top-p 0.95, seed 48150, and an explicit
64-token output budget. Each constrains output to an object containing its
fixture name and `ok: true`. Twenty-four outputs complete correctly. The PNG
orientation-2 and orientation-5 requests produce repeated legal JSON whitespace
and terminate with `finish_reason: length` at 64 tokens before completing the
object. Both failures recur when the complete fixture set runs without other
requests. Their HTTP status is 200, their SSE streams contain no runtime error,
and the service remains healthy. The complete output check therefore passes
28 of 30 cases, including malformed-input rejections.

Image decoding/admission and structured-output completion are separate results.
The whitespace loops remain an open output behavior; these checks do not
identify their cause or establish semantic vision accuracy. The request budgets
and inputs remain unchanged.
