function Parse-IFF {
    param (
        [string]$Path,
        [byte[]]$bytes,
        [int]$Offset = 0,
        [int]$EndOffset = -1,
        [int]$Indent = 0
    )
    if ($null -eq $bytes) { $bytes = [System.IO.File]::ReadAllBytes($Path) }
    if ($EndOffset -eq -1) { $EndOffset = $bytes.Length }
    
    while ($Offset + 8 -le $EndOffset) {
        $tag = [System.Text.Encoding]::ASCII.GetString($bytes, $Offset, 4)
        if ($tag -match "[^ -~]") { break }
        
        $len = [Net.IPAddress]::NetworkToHostOrder([BitConverter]::ToInt32($bytes, $Offset + 4))
        # Use a more lenient end check for nested blobs
        if ($len -lt 0) { break }

        $dispTag = $tag
        $subType = ""
        $typeTags = @("FORM", "LIST", "CAT ")
        if ($typeTags -contains $tag) {
            $subType = [System.Text.Encoding]::ASCII.GetString($bytes, $Offset + 8, 4)
            $dispTag = "$($tag):$($subType)"
        }

        Write-Host ("  " * $Indent + "[$Offset] $dispTag (len: $len)")
        
        # Look for strings in ANY chunk. Strings in these files are often null-terminated.
        if ($len -gt 4 -and $len -lt 1024) {
            $foundStrings = @()
            $currentStr = ""
            for ($i = 0; $i -lt $len; $i++) {
                $b = $bytes[$Offset + 8 + $i]
                if ($b -ge 32 -and $b -le 126) {
                    $currentStr += [char]$b
                } else {
                    if ($currentStr.Length -ge 4) { $foundStrings += $currentStr }
                    $currentStr = ""
                }
            }
            if ($currentStr.Length -ge 4) { $foundStrings += $currentStr }
            foreach ($s in $foundStrings) {
                if ($s -match "\.(tga|dds|iff|sht|eft|vsh|psh)$" -or $s.Length -gt 10) {
                     Write-Host ("  " * ($Indent + 1) + "String: $s")
                }
            }
        }

        if ($typeTags -contains $tag) {
            Parse-IFF -Path $Path -bytes $bytes -Offset ($Offset + 12) -EndOffset ($Offset + 8 + $len) -Indent ($Indent + 1)
        }
        else {
            # Try to find nested FORMs that might be embedded in other chunks (like EFCT DATA)
            if ($len -ge 12) {
                # Look for 'FORM' starting at Offset + 8 (the typical content start)
                $checkTag = [System.Text.Encoding]::ASCII.GetString($bytes, $Offset + 8, 4)
                if ($typeTags -contains $checkTag) {
                     Parse-IFF -Path $Path -bytes $bytes -Offset ($Offset + 8) -EndOffset ($Offset + 8 + $len) -Indent ($Indent + 1)
                }
            }
        }
        
        $Offset += 8 + $len
        if ($len % 2 -ne 0) { $Offset++ } 
    }
}

$files = @(
    "C:\Users\Hardy\Documents\naboo chain\shader\thed_brick_medb_asb13.sht",
    "C:\Users\Hardy\Documents\naboo chain\shader\thed_brick_largea_asb13.sht",
    "C:\Users\Hardy\Documents\naboo chain\effect\a_simple_bump.eft"
)

foreach ($f in $files) {
    Write-Host "`n--- Parsing $f ---" -ForegroundColor Cyan
    Parse-IFF -Path $f
}
