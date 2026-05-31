function Parse-IFF {
    param([byte[]]$bytes, [string]$path = "")
    $pos = 0
    while ($pos + 8 -le $bytes.Length) {
        $tag = [System.Text.Encoding]::ASCII.GetString($bytes, $pos, 4)
        $sizeBytes = $bytes[($pos + 4)..($pos + 7)]
        if ([BitConverter]::IsLittleEndian) { [Array]::Reverse($sizeBytes) }
        $size = [BitConverter]::ToUInt32($sizeBytes, 0)
        
        $currentPath = if ($path) { "$path/$tag" } else { $tag }
        $payloadStart = $pos + 8
        $payloadEnd = $pos + 7 + $size
        
        if ($tag -eq "FORM") {
            $type = [System.Text.Encoding]::ASCII.GetString($bytes, $payloadStart, 4)
            $subBytes = $bytes[($payloadStart + 4)..$payloadEnd]
            Parse-IFF -bytes $subBytes -path "$currentPath($type)"
        } else {
            $matchingTags = @("TCSS", "OPTN", "SCAP")
            $parentMatch = ($path -like "*(TCSS)*") -or ($path -like "*(OPTN)*") -or ($path -like "*(SCAP)*")
            
            if ($matchingTags -contains $tag -or ($tag -eq "DATA" -and $parentMatch)) {
                $payload = $bytes[$payloadStart..$payloadEnd]
                Write-Host "Chunk: $tag Path: $currentPath Size: $size"
                if ($payload.Length -gt 0) {
                    $hex = [System.BitConverter]::ToString($payload)
                    Write-Host "  Hex: $hex"
                    if ($payload.Length -ge 4) {
                        for ($i = 0; $i -le $payload.Length - 4; $i += 4) {
                            $leInt = [BitConverter]::ToInt32($payload, $i)
                            $leFloat = [BitConverter]::ToSingle($payload, $i)
                            $beBytes = $payload[$i..($i+3)]; [Array]::Reverse($beBytes)
                            $beInt = [BitConverter]::ToInt32($beBytes, 0)
                            $beFloat = [BitConverter]::ToSingle($beBytes, 0)
                            Write-Host "  [$i] LE: Int=$leInt, Float=$leFloat | BE: Int=$beInt, Float=$beFloat"
                        }
                    }
                }
            }
        }
        $pos += 8 + $size
        if ($size % 2 -ne 0) { $pos++ }
    }
}

$files = @("C:\Users\Hardy\Documents\naboo chain\shader\thed_brick_medb_asb13.sht", "C:\Users\Hardy\Documents\naboo chain\effect\a_simple_bump.eft")
foreach ($f in $files) {
    if (Test-Path $f) {
        Write-Host "`n--- Processing $f ---"
        $b = [System.IO.File]::ReadAllBytes($f)
        Parse-IFF -bytes $b
    }
}
