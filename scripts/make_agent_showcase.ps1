Add-Type -AssemblyName System.Drawing

$W = 1280
$H = 800
$bmp = New-Object System.Drawing.Bitmap($W, $H)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAlias

function New-SolidBrush([string]$hex) {
    $hex = $hex.TrimStart('#')
    return New-Object System.Drawing.SolidBrush(
        [System.Drawing.Color]::FromArgb(
            [Convert]::ToInt32($hex.Substring(0,2), 16),
            [Convert]::ToInt32($hex.Substring(2,2), 16),
            [Convert]::ToInt32($hex.Substring(4,2), 16)
        )
    )
}

function New-Pen([string]$hex, [float]$width) {
    return New-Object System.Drawing.Pen(
        [System.Drawing.Color]::FromArgb(
            [Convert]::ToInt32($hex.Substring(1,2), 16),
            [Convert]::ToInt32($hex.Substring(3,2), 16),
            [Convert]::ToInt32($hex.Substring(5,2), 16)
        ),
        $width
    )
}

function Draw-RoundRect(
    [System.Drawing.Graphics]$g,
    [System.Drawing.Brush]$brush,
    [float]$x,
    [float]$y,
    [float]$w,
    [float]$h,
    [float]$r
) {
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $d = $r * 2
    $path.AddArc($x, $y, $d, $d, 180, 90)
    $path.AddArc($x + $w - $d, $y, $d, $d, 270, 90)
    $path.AddArc($x + $w - $d, $y + $h - $d, $d, $d, 0, 90)
    $path.AddArc($x, $y + $h - $d, $d, $d, 90, 90)
    $path.CloseFigure()
    $g.FillPath($brush, $path)
    $path.Dispose()
}

function Draw-RoundRectOutline(
    [System.Drawing.Graphics]$g,
    [System.Drawing.Pen]$pen,
    [float]$x,
    [float]$y,
    [float]$w,
    [float]$h,
    [float]$r
) {
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $d = $r * 2
    $path.AddArc($x, $y, $d, $d, 180, 90)
    $path.AddArc($x + $w - $d, $y, $d, $d, 270, 90)
    $path.AddArc($x + $w - $d, $y + $h - $d, $d, $d, 0, 90)
    $path.AddArc($x, $y + $h - $d, $d, $d, 90, 90)
    $path.CloseFigure()
    $g.DrawPath($pen, $path)
    $path.Dispose()
}

function Draw-Text(
    [System.Drawing.Graphics]$g,
    [string]$text,
    [string]$fontName,
    [float]$size,
    [string]$colorHex,
    [float]$x,
    [float]$y,
    [System.Drawing.FontStyle]$style = [System.Drawing.FontStyle]::Regular
) {
    $font = New-Object System.Drawing.Font($fontName, $size, $style)
    $brush = New-SolidBrush $colorHex
    $g.DrawString($text, $font, $brush, $x, $y)
    $brush.Dispose()
    $font.Dispose()
}

function Draw-CenteredText(
    [System.Drawing.Graphics]$g,
    [string]$text,
    [string]$fontName,
    [float]$size,
    [string]$colorHex,
    [float]$centerX,
    [float]$y,
    [System.Drawing.FontStyle]$style = [System.Drawing.FontStyle]::Regular
) {
    $font = New-Object System.Drawing.Font($fontName, $size, $style)
    $brush = New-SolidBrush $colorHex
    $sf = New-Object System.Drawing.StringFormat
    $sf.Alignment = [System.Drawing.StringAlignment]::Center
    $rect = New-Object System.Drawing.RectangleF(($centerX - 400), $y, 800, ($size + 12))
    $g.DrawString($text, $font, $brush, $rect, $sf)
    $sf.Dispose()
    $brush.Dispose()
    $font.Dispose()
}

function Draw-SearchIcon([System.Drawing.Graphics]$g, [float]$x, [float]$y, [float]$s, [string]$color) {
    $pen = New-Pen $color 2.2
    $g.DrawEllipse($pen, $x, $y, $s * 0.68, $s * 0.68)
    $g.DrawLine($pen, ($x + $s * 0.5), ($y + $s * 0.5), ($x + $s * 0.92), ($y + $s * 0.92))
    $pen.Dispose()
}

function Draw-DocIcon([System.Drawing.Graphics]$g, [float]$x, [float]$y, [float]$s, [string]$color) {
    $pen = New-Pen $color 2.0
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $rect = New-Object System.Drawing.RectangleF -ArgumentList ([float]$x, [float]$y, [float]($s * 0.72), [float]$s)
    $path.AddRectangle($rect)
    $g.DrawPath($pen, $path)
    $g.DrawLine($pen, ($x + $s * 0.48), $y, ($x + $s * 0.48), ($y + $s * 0.34))
    $g.DrawLine($pen, ($x + $s * 0.48), ($y + $s * 0.34), ($x + $s * 0.72), ($y + $s * 0.34))
    $g.DrawLine($pen, ($x + $s * 0.2), ($y + $s * 0.52), ($x + $s * 0.52), ($y + $s * 0.52))
    $g.DrawLine($pen, ($x + $s * 0.2), ($y + $s * 0.72), ($x + $s * 0.52), ($y + $s * 0.72))
    $path.Dispose()
    $pen.Dispose()
}

function Draw-BookIcon([System.Drawing.Graphics]$g, [float]$x, [float]$y, [float]$s, [string]$color) {
    $pen = New-Pen $color 2.0
    $g.DrawArc($pen, $x, $y, ($s * 0.5), $s, 90, 180)
    $g.DrawArc($pen, ($x + $s * 0.5), $y, ($s * 0.5), $s, 270, 180)
    $g.DrawLine($pen, ($x + $s * 0.25), ($y + $s * 0.12), ($x + $s * 0.25), ($y + $s * 0.88))
    $g.DrawLine($pen, ($x + $s * 0.75), ($y + $s * 0.12), ($x + $s * 0.75), ($y + $s * 0.88))
    $pen.Dispose()
}

function Draw-QuoteIcon([System.Drawing.Graphics]$g, [float]$x, [float]$y, [float]$s, [string]$color) {
    $pen = New-Pen $color 2.2
    $g.DrawArc($pen, $x, $y, $s, $s, 180, 180)
    $g.DrawLine($pen, ($x + $s * 0.2), ($y + $s * 0.42), ($x + $s * 0.8), ($y + $s * 0.42))
    $g.DrawLine($pen, ($x + $s * 0.2), ($y + $s * 0.62), ($x + $s * 0.64), ($y + $s * 0.62))
    $pen.Dispose()
}

function Draw-GearIcon([System.Drawing.Graphics]$g, [float]$x, [float]$y, [float]$s, [string]$color) {
    $pen = New-Pen $color 2.0
    $cx = $x + $s * 0.5
    $cy = $y + $s * 0.5
    $g.DrawEllipse($pen, ($cx - $s * 0.18), ($cy - $s * 0.18), ($s * 0.36), ($s * 0.36))
    for ($i = 0; $i -lt 8; $i++) {
        $a = [Math]::PI * $i / 4
        $x1 = $cx + [Math]::Cos($a) * $s * 0.30
        $y1 = $cy + [Math]::Sin($a) * $s * 0.30
        $x2 = $cx + [Math]::Cos($a) * $s * 0.44
        $y2 = $cy + [Math]::Sin($a) * $s * 0.44
        $g.DrawLine($pen, [float]$x1, [float]$y1, [float]$x2, [float]$y2)
    }
    $pen.Dispose()
}

function Draw-HelpIcon([System.Drawing.Graphics]$g, [float]$x, [float]$y, [float]$s, [string]$color) {
    $pen = New-Pen $color 2.0
    $g.DrawEllipse($pen, $x, $y, $s, $s)
    $g.DrawArc($pen, ($x + $s * 0.25), ($y + $s * 0.22), ($s * 0.5), ($s * 0.5), 180, 270)
    $g.DrawLine($pen, ($x + $s * 0.5), ($y + $s * 0.72), ($x + $s * 0.5), ($y + $s * 0.84))
    $pen.Dispose()
}

function Draw-PlusIcon([System.Drawing.Graphics]$g, [float]$x, [float]$y, [float]$s, [string]$color) {
    $pen = New-Pen $color 2.2
    $g.DrawLine($pen, ($x + $s * 0.5), $y, ($x + $s * 0.5), ($y + $s))
    $g.DrawLine($pen, $x, ($y + $s * 0.5), ($x + $s), ($y + $s * 0.5))
    $pen.Dispose()
}

function Draw-ResearchIcon([System.Drawing.Graphics]$g, [float]$x, [float]$y, [float]$s, [string]$color) {
    $pen = New-Pen $color 2.0
    $g.DrawEllipse($pen, ($x + $s * 0.18), ($y + $s * 0.08), ($s * 0.64), ($s * 0.64))
    $g.DrawLine($pen, ($x + $s * 0.82), ($y + $s * 0.76), ($x + $s * 0.96), ($y + $s * 0.96))
    $g.DrawLine($pen, ($x + $s * 0.4), ($y + $s * 0.42), ($x + $s * 0.6), ($y + $s * 0.42))
    $pen.Dispose()
}

# 背景
$bgBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
    (New-Object System.Drawing.Point(0, 0)),
    (New-Object System.Drawing.Point(0, $H)),
    ([System.Drawing.Color]::FromArgb(247, 248, 250)),
    ([System.Drawing.Color]::FromArgb(255, 255, 255))
)
$g.FillRectangle($bgBrush, 0, 0, $W, $H)
$bgBrush.Dispose()

# 侧边栏
$sidebarBrush = New-SolidBrush '#FAFAFB'
$g.FillRectangle($sidebarBrush, 0, 0, 280, $H)
$sidebarBrush.Dispose()
$sidebarBorder = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(229, 231, 235), 1)
$g.DrawLine($sidebarBorder, 279, 0, 279, $H)
$sidebarBorder.Dispose()

# Logo
Draw-RoundRect $g (New-SolidBrush '#4F46E5') 22 22 36 36 10
Draw-CenteredText $g '知' 'Microsoft YaHei' 15 '#FFFFFF' 40 29 ([System.Drawing.FontStyle]::Bold)
Draw-Text $g 'KnowResearch' 'Microsoft YaHei' 15 '#111827' 68 30 ([System.Drawing.FontStyle]::Bold)
Draw-Text $g '科研知识助手' 'Microsoft YaHei' 9 '#9CA3AF' 68 51

# 新对话按钮
Draw-RoundRect $g (New-SolidBrush '#4F46E5') 20 82 240 44 10
Draw-PlusIcon $g 34 94 20 '#FFFFFF'
Draw-Text $g '新对话' 'Microsoft YaHei' 12 '#FFFFFF' 64 94 ([System.Drawing.FontStyle]::Bold)

# 底部导航
Draw-GearIcon $g 26 688 22 '#6B7280'
Draw-Text $g '设置' 'Microsoft YaHei' 11 '#374151' 60 690
Draw-HelpIcon $g 26 728 22 '#6B7280'
Draw-Text $g '帮助' 'Microsoft YaHei' 11 '#374151' 60 730

# 顶部右侧头像
Draw-RoundRect $g (New-SolidBrush '#EEF2FF') 1212 24 36 36 18
Draw-CenteredText $g '研' 'Microsoft YaHei' 13 '#4338CA' 1230 31 ([System.Drawing.FontStyle]::Bold)

# 中间 Hero
Draw-RoundRect $g (New-SolidBrush '#4F46E5') 608 178 64 64 16
Draw-CenteredText $g '知' 'Microsoft YaHei' 24 '#FFFFFF' 640 192 ([System.Drawing.FontStyle]::Bold)

Draw-CenteredText $g '知研 KnowResearch' 'Microsoft YaHei' 26 '#111827' 640 270 ([System.Drawing.FontStyle]::Bold)
Draw-CenteredText $g '你的科研知识库 Agent' 'Microsoft YaHei' 14 '#6B7280' 640 316

# 搜索框
Draw-RoundRect $g (New-SolidBrush '#FFFFFF') 400 368 480 54 27
Draw-RoundRectOutline $g (New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(209,213,219),1)) 400 368 480 54 27
Draw-SearchIcon $g 424 386 22 '#6B7280'
Draw-Text $g '搜索你的科研知识库…' 'Microsoft YaHei' 12 '#9CA3AF' 458 384

# 功能卡片
$cards = @(
    @('论文解析', 'doc', '#4F46E5'),
    @('混合检索', 'research', '#0F766E'),
    @('引用溯源', 'quote', '#B45309'),
    @('文献调研', 'book', '#7C3AED')
)
$cardY = 476
$cardH = 70
$cardW = 100
$gap = 28
$startX = [float](($W - ($cardW * 4 + $gap * 3)) / 2)
for ($i = 0; $i -lt $cards.Count; $i++) {
    $cx = $startX + $i * ($cardW + $gap)
    Draw-RoundRect $g (New-SolidBrush '#FFFFFF') $cx $cardY $cardW $cardH 12
    Draw-RoundRectOutline $g (New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(229,231,235),1)) $cx $cardY $cardW $cardH 12
    $iconX = $cx + (($cardW - 26) / 2)
    if ($cards[$i][1] -eq 'doc') { Draw-DocIcon $g $iconX ($cardY + 12) 26 $cards[$i][2] }
    elseif ($cards[$i][1] -eq 'research') { Draw-ResearchIcon $g $iconX ($cardY + 12) 26 $cards[$i][2] }
    elseif ($cards[$i][1] -eq 'quote') { Draw-QuoteIcon $g $iconX ($cardY + 12) 26 $cards[$i][2] }
    else { Draw-BookIcon $g $iconX ($cardY + 12) 26 $cards[$i][2] }
    Draw-CenteredText $g $cards[$i][0] 'Microsoft YaHei' 10 '#374151' ($cx + $cardW / 2) ($cardY + 46) ([System.Drawing.FontStyle]::Bold)
}

# 底部信息
Draw-CenteredText $g '本地知识库 · 隐私优先' 'Microsoft YaHei' 11 '#9CA3AF' 640 726

$g.Dispose()

$outDir = Join-Path $PSScriptRoot '..\docs\assets'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outPath = Join-Path $outDir 'agent_showcase.png'
$bmp.Save($outPath, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()
Write-Output $outPath
