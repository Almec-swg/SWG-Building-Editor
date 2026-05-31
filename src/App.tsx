import { useCallback, useEffect, useMemo, useRef, useState, memo } from 'react'
import * as THREE from 'three'
import { DDSLoader } from 'three/examples/jsm/loaders/DDSLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { TGALoader } from 'three/examples/jsm/loaders/TGALoader.js'
import { extractBuildingObjectsFromIff, parseIff } from './iffParser'
import { buildFileTree } from './fileTree'
import type { FileTreeNode } from './fileTree'
import {
  normalizeSwgPath,
  resolveTemplateVisual,
  type RepositorySourceFile,
  type ResolvedTemplateVisual,
  extractMeshFromAsset,
  getShaderDebugChunks,
  getEffectDebugChunks,
  getShaderRenderProps,
} from './previewResolver'
import { extractTreRecord, parseTreArchive, type TreCompression, type TreRecordMeta } from './treParser'
import './App.css'

type AssetKind = 'building' | 'object' | 'appearance'

interface AssetRecord {
  id: string
  name: string
  kind: AssetKind
  size: number
  lastModified: number
  relativePath: string
  source: 'loose' | 'tre'
}

interface IndexedFileEntry {
  file: File
  relativePath: string
}

interface TreIndexedEntry {
  id: string
  treName: string
  treFile: File
  path: string
  name: string
  kind: AssetKind
  size: number
  record: TreRecordMeta
  dataCompression: TreCompression
}

interface TreLookupRecord {
  treName: string
  treFile: File
  record: TreRecordMeta
}

interface PlacedObject {
  id: string
  name: string
  sourceAssetId: string
  templatePath?: string
  x: number
  z: number
  y: number
  yaw: number
  scale: number
  appearanceAssetId?: string
}

interface BuildingDocument {
  id: string
  name: string
  rootTemplatePath?: string
  sourceAssetId?: string
  width: number
  depth: number
  height: number
  objects: PlacedObject[]
}

type FileTreeItem =
  | { treePath: string; source: 'loaded'; assetId: string }
  | { treePath: string; source: 'indexed'; relativePath: string }
  | { treePath: string; source: 'tre'; treEntryId: string }

type ResolvedVisualMap = Record<string, ResolvedTemplateVisual | undefined>
type ResolvedTextureMap = Record<string, {
  url: string
  texturePath: string
  alternateUrl?: string
  alternateTexturePath?: string
  textureAddressU?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce'
  textureAddressV?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce'
  textureMipmapFilter?: 'none' | 'point' | 'linear' | 'anisotropic' | 'flatcubic' | 'gaussiancubic'
  textureMinificationFilter?: 'none' | 'point' | 'linear' | 'anisotropic' | 'flatcubic' | 'gaussiancubic'
  textureMagnificationFilter?: 'none' | 'point' | 'linear' | 'anisotropic' | 'flatcubic' | 'gaussiancubic'
  normalUrl?: string
  normalTexturePath?: string
  normalTextureAddressU?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce'
  normalTextureAddressV?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce'
  secondaryUrl?: string
  secondaryTexturePath?: string
  secondaryTextureAddressU?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce'
  secondaryTextureAddressV?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce'
  primaryUvSetIndex?: number
  secondaryUvSetIndex?: number
} | undefined>

interface SurfacePickMetadata {
  scope: 'root' | 'object'
  buildingName?: string
  objectId?: string
  objectName?: string
  templatePath?: string
  meshPath?: string
  appearancePath?: string
  shaderPath?: string
  effectPath?: string
  sourceLabel?: string
  textureLookupKey?: string
  partDomain?: 'exterior' | 'interior'
  partIndex?: number
  primaryUvSetIndex?: number
  secondaryUvSetIndex?: number
  baseTexturePath?: string
  normalTexturePath?: string
  detailTexturePath?: string
  baseAddressU?: string
  baseAddressV?: string
  normalAddressU?: string
  normalAddressV?: string
  detailAddressU?: string
  detailAddressV?: string
  chunkTrace?: string
  shaderDebugChunks?: string[]  // Shader IFF structure for debugging
  effectDebugChunks?: string[]   // Effect (.eft) IFF structure for debugging
  uvCalibrationTransform?: string
  uvCalibrationScaleU?: number
  uvCalibrationScaleV?: number
  uvCalibrationTargetSource?: string
}

function buildPartTextureKey(templatePath: string, partIndex: number): string {
  return `${templatePath}#part:${partIndex}`
}

const kb = 1024
const knownAppearanceExtensions = new Set([
  '.pob',
  '.apt',
  '.msh',
  '.lod',
  '.cmp',
  '.flr',
  '.dds',
  '.tga',
  '.trt',
  '.sat',
  '.sht',
])

// treev5.lua repository order from SIE (later entries override earlier ones).
const sieTreeV5Order = [
  'bottom.tre',
  'data_music_00.tre',
  'data_sample_00.tre',
  'data_sample_01.tre',
  'data_sample_02.tre',
  'data_sample_03.tre',
  'data_sample_04.tre',
  'data_texture_00.tre',
  'data_texture_01.tre',
  'data_texture_02.tre',
  'data_texture_03.tre',
  'data_texture_04.tre',
  'data_texture_05.tre',
  'data_texture_06.tre',
  'data_texture_07.tre',
  'data_other_00.tre',
  'data_animation_00.tre',
  'data_skeletal_mesh_00.tre',
  'data_skeletal_mesh_01.tre',
  'data_static_mesh_00.tre',
  'data_static_mesh_01.tre',
  'patch_00.tre',
  'patch_01.tre',
  'patch_02.tre',
  'patch_03.tre',
  'patch_04.tre',
  'patch_05.tre',
  'patch_06.tre',
  'patch_07.tre',
  'patch_08.tre',
  'patch_09.tre',
  'patch_10.tre',
  'data_sku1_00.tre',
  'data_sku1_01.tre',
  'data_sku1_02.tre',
  'data_sku1_03.tre',
  'data_sku1_04.tre',
  'data_sku1_05.tre',
  'patch_11_00.tre',
  'patch_11_01.tre',
  'data_sku1_06.tre',
  'patch_11_02.tre',
  'data_sku1_07.tre',
  'patch_11_03.tre',
  'patch_12_00.tre',
  'patch_sku1_12_00.tre',
  'patch_13_00.tre',
  'patch_14_00.tre',
  'patch_15_00.tre',
  'patch_15_01.tre',
  'patch_15_02.tre',
  'patch_16_00.tre',
  'patch_17_00.tre',
  'patch_sku1_14_00.tre',
  'default_patch.tre',
]

const sieTreeV5Priority = new Map(
  sieTreeV5Order.map((name, index) => [name.toLowerCase(), index]),
)

interface FileSystemDirectoryHandleLike {
  kind: 'directory'
  name: string
  entries(): AsyncIterable<[string, FileSystemHandleLike]>
}

interface FileSystemFileHandleLike {
  kind: 'file'
  name: string
  getFile(): Promise<File>
}

type FileSystemHandleLike = FileSystemDirectoryHandleLike | FileSystemFileHandleLike

interface WindowWithDirectoryPicker extends Window {
  showDirectoryPicker?: () => Promise<FileSystemDirectoryHandleLike>
}

function newId(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function formatBytes(bytes: number): string {
  if (bytes < kb) return `${bytes} B`
  return `${(bytes / kb).toFixed(1)} KB`
}

function extensionOf(fileName: string): string {
  const dot = fileName.lastIndexOf('.')
  if (dot < 0) return ''
  return fileName.slice(dot).toLowerCase()
}

function inferAssetKind(fileName: string, relativePath: string): AssetKind | null {
  const lowerName = fileName.toLowerCase()
  const lowerPath = relativePath.toLowerCase().replace(/\\/g, '/')
  const ext = extensionOf(lowerName)

  if (knownAppearanceExtensions.has(ext)) return 'appearance'

  if (ext !== '.iff') return null
  if (lowerPath.includes('/building/') || lowerPath.includes('/footprint/')) return 'building'
  if (lowerPath.includes('/appearance/')) return 'appearance'
  if (lowerPath.includes('/object/') || lowerPath.includes('/interiorlayout/')) return 'object'
  if (lowerName.includes('building')) return 'building'
  return 'object'
}

function treArchivePriority(treName: string): number {
  const lower = treName.toLowerCase()
  const exact = sieTreeV5Priority.get(lower)
  if (exact !== undefined) return exact

  // Fallback ranking for non-standard names.
  const patchMatch = lower.match(/^patch(?:_sku\d+)?_(\d+)(?:_(\d+))?\.tre$/)
  if (patchMatch) {
    const major = Number(patchMatch[1] ?? '0')
    const minor = Number(patchMatch[2] ?? '0')
    return sieTreeV5Order.length + 50000 + major * 100 + minor
  }

  if (lower.startsWith('data_')) return sieTreeV5Order.length + 1000
  return sieTreeV5Order.length + 100
}

async function indexTreFile(file: File): Promise<TreIndexedEntry[]> {
  const parsed = await parseTreArchive(file)
  const entries: TreIndexedEntry[] = []
  for (const record of parsed.records) {
    const path = record.name.toLowerCase()
    const name = path.split('/').pop() ?? path
    const kind = inferAssetKind(name, path)
    if (!kind) continue

    entries.push({
      id: newId('treEntry'),
      treName: file.name,
      treFile: file,
      path,
      name,
      kind,
      size: record.dataUncompressed,
      record,
      dataCompression: record.dataCompression,
    })
  }

  return entries
}

async function extractTreEntryFile(entry: TreIndexedEntry): Promise<File | null> {
  try {
    return await extractTreRecord(entry.treFile, entry.record)
  } catch {
    return null
  }
}

const PreviewCanvas = memo(function PreviewCanvas({
  building,
  selectedObjectId,
  resolvedVisuals,
  resolvedTextures,
  renderTextures,
  interiorAssist,
  interiorOpacity,
  roofCutaway,
  roofCutawayLevel,
  showPlaceholderObjects,
}: {
  building: BuildingDocument | undefined
  selectedObjectId: string | null
  resolvedVisuals: ResolvedVisualMap
  resolvedTextures: ResolvedTextureMap
  renderTextures: boolean
  interiorAssist: boolean
  interiorOpacity: number
  roofCutaway: boolean
  roofCutawayLevel: number
  showPlaceholderObjects: boolean
}) {
  const [compilingShaders, setCompilingShaders] = useState(false)
  const [pickedSurfaceDump, setPickedSurfaceDump] = useState('Click a rendered surface to dump texture binding data.')
  const [pickCopyStatus, setPickCopyStatus] = useState('')
  const containerRef = useRef<HTMLDivElement | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const previewGroupRef = useRef<THREE.Group | null>(null)
  const textureLoaderRef = useRef<THREE.TextureLoader | null>(null)
  const ddsLoaderRef = useRef<DDSLoader | null>(null)
  const tgaLoaderRef = useRef<TGALoader | null>(null)
  const textureCacheRef = useRef<Map<string, THREE.Texture>>(new Map())
  const textureAccessOrderRef = useRef<string[]>([])
  const textureSizeRef = useRef(0)
  const resolvedTextureAliasRef = useRef<Map<string, NonNullable<ResolvedTextureMap[string]>>>(new Map())
  const rejectedTexturePathsRef = useRef<Set<string>>(new Set())
  const uvFallbackLoggedRef = useRef<Set<string>>(new Set())
  const previewBuildEpochRef = useRef(0)
  const rootCutawayMaterialsRef = useRef<THREE.MeshPhongMaterial[]>([])
  const globalCutawayPlaneRef = useRef<THREE.Plane>(new THREE.Plane(new THREE.Vector3(0, -1, 0), 999999))
  const resolvedBuildingHeightRef = useRef<number>(20)
  const requestRenderRef = useRef<() => void>(() => {})
  const compileTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  // renderTexturesRef / interiorOpacityRef let the mesh-build effect read the latest values
  // without being in its deps. Changes are handled by the lightweight material-swap effect.
  const renderTexturesRef = useRef(renderTextures)
  renderTexturesRef.current = renderTextures
  const interiorOpacityRef = useRef(interiorOpacity)
  interiorOpacityRef.current = interiorOpacity
  const applyTextureToMaterialRef = useRef<(
    templatePath: string | undefined,
    material: THREE.MeshPhongMaterial,
    context: 'object' | 'root',
  ) => void>(() => {})
  const applyStoredShaderMaterialStateRef = useRef<(
    material: THREE.MeshPhongMaterial,
  ) => void>(() => {})
  const allMaterialsRef = useRef<Array<{
    material: THREE.MeshPhongMaterial
    templatePath: string | undefined
    context: 'object' | 'root'
  }>>([])
  const geometryCacheRef = useRef<Map<string, THREE.BufferGeometry>>(new Map())
  const MAX_TEXTURE_CACHE_SIZE = 256 * 1024 * 1024 // 256MB
  // Shared 1×1 white placeholder so USE_MAP is baked into all shader programs at creation.
  // Swapping in real textures later never changes this define → zero shader recompilation on color map load.
  const placeholderColorTexRef = useRef<THREE.DataTexture | null>(null)
  // Shared 1×1 flat-normal placeholder so USE_NORMALMAP is baked in from the start.
  // Toggling between this and a real normal map never changes the define → no shader recompile.
  const placeholderNormalTexRef = useRef<THREE.DataTexture | null>(null)

  const formatMode = (value?: string): string => value ?? 'n/a'

  const buildSurfaceDump = (
    hit: THREE.Intersection<THREE.Object3D>,
    meta: SurfacePickMetadata,
    allHits: THREE.Intersection<THREE.Object3D>[],
  ): string => {
    const now = new Date().toISOString()
    const point = hit.point
    const altTextureCandidates = (rawPath: string): string[] => {
      const lower = rawPath.toLowerCase()
      if (lower.endsWith('.dds')) return [lower, `${lower.slice(0, -4)}.tga`]
      if (lower.endsWith('.tga')) return [lower, `${lower.slice(0, -4)}.dds`]
      return [lower]
    }
    const object = hit.object as THREE.Mesh
    const rawMaterial = object.material
    const hitMaterial = Array.isArray(rawMaterial)
      ? rawMaterial.find((entry) => entry instanceof THREE.MeshPhongMaterial)
      : rawMaterial
    const phongMaterial = hitMaterial instanceof THREE.MeshPhongMaterial ? hitMaterial : undefined
    const materialColorPath = ((phongMaterial?.map?.userData as { swgTexturePath?: string } | undefined)?.swgTexturePath)
    const materialNormalPath = ((phongMaterial?.normalMap?.userData as { swgTexturePath?: string } | undefined)?.swgTexturePath)

    const directTextureMeta = meta.textureLookupKey
      ? (resolvedTextureAliasRef.current.get(meta.textureLookupKey) ?? resolvedTextures[meta.textureLookupKey])
      : undefined
    const textureMeta = directTextureMeta ?? (() => {
      const expectedTexturePath = meta.baseTexturePath
      if (!expectedTexturePath) return undefined
      const candidates = altTextureCandidates(expectedTexturePath)
      const candidateFiles = new Set(candidates.map((value) => value.split('/').pop() ?? value))
      return Object.values(resolvedTextures).find((entry) => {
        if (!entry?.texturePath) return false
        const entryPath = entry.texturePath.toLowerCase()
        if (candidates.includes(entryPath)) return true
        const entryFile = entryPath.split('/').pop() ?? entryPath
        return candidateFiles.has(entryFile)
      })
    })()
    const resolvedBy = directTextureMeta
      ? 'lookupKey'
      : textureMeta
        ? 'texturePathFallback'
        : materialColorPath || materialNormalPath
          ? 'materialMap'
        : 'n/a'
    const geometry = object.geometry as THREE.BufferGeometry | undefined
    const positionAttr = geometry?.getAttribute('position') as THREE.BufferAttribute | undefined
    const indexAttr = geometry?.getIndex()
    const uvAttr = geometry?.getAttribute('uv') as THREE.BufferAttribute | undefined
    const uv2Attr = geometry?.getAttribute('uv2') as THREE.BufferAttribute | undefined
    const triCount = indexAttr
      ? Math.floor(indexAttr.count / 3)
      : positionAttr
        ? Math.floor(positionAttr.count / 3)
        : 0

    const computeUvStats = (attr?: THREE.BufferAttribute): string => {
      if (!attr || attr.itemSize < 2 || attr.count <= 0) return 'n/a'
      let minU = Number.POSITIVE_INFINITY
      let minV = Number.POSITIVE_INFINITY
      let maxU = Number.NEGATIVE_INFINITY
      let maxV = Number.NEGATIVE_INFINITY
      for (let i = 0; i < attr.count; i += 1) {
        const u = attr.getX(i)
        const v = attr.getY(i)
        if (!Number.isFinite(u) || !Number.isFinite(v)) continue
        if (u < minU) minU = u
        if (v < minV) minV = v
        if (u > maxU) maxU = u
        if (v > maxV) maxV = v
      }
      if (!Number.isFinite(minU) || !Number.isFinite(minV) || !Number.isFinite(maxU) || !Number.isFinite(maxV)) {
        return 'n/a'
      }
      return `[${minU.toFixed(6)}, ${minV.toFixed(6)}] -> [${maxU.toFixed(6)}, ${maxV.toFixed(6)}]`
    }

    const overlapHits = allHits
      .filter((entry) => Math.abs(entry.distance - hit.distance) <= 0.02)
      .filter((entry) => Boolean((entry.object as THREE.Object3D).userData?.surfacePick))
      .slice(0, 10)

    const overlapLines = overlapHits.map((entry, i) => {
      const overlapMeta = entry.object.userData.surfacePick as SurfacePickMetadata | undefined
      return `overlap[${i}]: d=${entry.distance.toFixed(6)} part=${overlapMeta?.partIndex ?? 'n/a'} tex=${overlapMeta?.baseTexturePath ?? 'n/a'} shader=${overlapMeta?.shaderPath ?? 'n/a'} trace=${overlapMeta?.chunkTrace ?? 'n/a'}`
    })

    const lines = [
      '=== SWG Surface Pick Dump ===',
      `pickedAt: ${now}`,
      `scope: ${meta.scope}`,
      `building: ${meta.buildingName ?? 'n/a'}`,
      `objectId: ${meta.objectId ?? 'n/a'}`,
      `objectName: ${meta.objectName ?? 'n/a'}`,
      `templatePath: ${meta.templatePath ?? 'n/a'}`,
      `meshPath: ${meta.meshPath ?? 'n/a'}`,
      `appearancePath: ${meta.appearancePath ?? 'n/a'}`,
      `shaderPath: ${meta.shaderPath ?? 'n/a'}`,
      `effectPath: ${meta.effectPath ?? 'n/a'}`,
      `sourceLabel: ${meta.sourceLabel ?? 'n/a'}`,
      `partIndex: ${meta.partIndex ?? 'n/a'}`,
      `primaryUvSetIndex: ${meta.primaryUvSetIndex ?? 'n/a'}`,
      `secondaryUvSetIndex: ${meta.secondaryUvSetIndex ?? 'n/a'}`,
      `baseTexturePath: ${meta.baseTexturePath ?? 'n/a'}`,
      `normalTexturePath: ${meta.normalTexturePath ?? 'n/a'}`,
      `detailTexturePath: ${meta.detailTexturePath ?? 'n/a'}`,
      `baseAddress: [${formatMode(meta.baseAddressU)}, ${formatMode(meta.baseAddressV)}]`,
      `normalAddress: [${formatMode(meta.normalAddressU)}, ${formatMode(meta.normalAddressV)}]`,
      `detailAddress: [${formatMode(meta.detailAddressU)}, ${formatMode(meta.detailAddressV)}]`,
      `chunkTrace: ${meta.chunkTrace ?? 'n/a'}`,
      `textureLookupKey: ${meta.textureLookupKey ?? 'n/a'}`,
      `partDomain: ${meta.partDomain ?? 'n/a'}`,
      `resolvedBy: ${resolvedBy}`,
      `resolvedTextureUrl: ${textureMeta?.url ?? materialColorPath ?? 'n/a'}`,
      `resolvedNormalUrl: ${textureMeta?.normalUrl ?? materialNormalPath ?? 'n/a'}`,
      `resolvedSecondaryUrl: ${textureMeta?.secondaryUrl ?? 'n/a'}`,
      `resolvedTexturePath: ${textureMeta?.texturePath ?? materialColorPath ?? 'n/a'}`,
      `resolvedNormalPath: ${textureMeta?.normalTexturePath ?? materialNormalPath ?? 'n/a'}`,
      `resolvedSecondaryPath: ${textureMeta?.secondaryTexturePath ?? 'n/a'}`,
      `material.transparent: ${phongMaterial?.transparent ?? 'n/a'}`,
      `material.opacity: ${phongMaterial ? phongMaterial.opacity.toFixed(6) : 'n/a'}`,
      `material.alphaTest: ${phongMaterial ? phongMaterial.alphaTest.toFixed(6) : 'n/a'}`,
      `faceIndex: ${hit.faceIndex ?? 'n/a'}`,
      `distance: ${Number.isFinite(hit.distance) ? hit.distance.toFixed(6) : 'n/a'}`,
      `hitPoint: [${point.x.toFixed(6)}, ${point.y.toFixed(6)}, ${point.z.toFixed(6)}]`,
      `meshVertexCount: ${positionAttr?.count ?? 0}`,
      `meshTriangleCount: ${triCount}`,
      `uvCount: ${uvAttr?.count ?? 0}`,
      `uv2Count: ${uv2Attr?.count ?? 0}`,
      `uvRange: ${computeUvStats(uvAttr)}`,
      `uv2Range: ${computeUvStats(uv2Attr)}`,
      (() => {
        if (!positionAttr) return `meshBounds: n/a`
        let minX = Infinity, minY = Infinity, minZ = Infinity
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity
        for (let i = 0; i < positionAttr.count; i++) {
          const x = positionAttr.getX(i), y = positionAttr.getY(i), z = positionAttr.getZ(i)
          if (x < minX) minX = x; if (x > maxX) maxX = x
          if (y < minY) minY = y; if (y > maxY) maxY = y
          if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
        }
        const dx = (maxX - minX).toFixed(3), dy = (maxY - minY).toFixed(3), dz = (maxZ - minZ).toFixed(3)
        return `meshBounds: [${minX.toFixed(3)},${minY.toFixed(3)},${minZ.toFixed(3)}] -> [${maxX.toFixed(3)},${maxY.toFixed(3)},${maxZ.toFixed(3)}] (dX=${dx} dY=${dy} dZ=${dz})`
      })(),
      (() => {
        // Show UV + XYZ for the 3 vertices of the hit triangle
        const fi = hit.faceIndex
        if (fi == null || !uvAttr || !positionAttr) return 'hitFace: n/a'
        const idx = geometry?.index
        let a: number, b: number, c: number
        if (idx) {
          a = idx.getX(fi * 3); b = idx.getX(fi * 3 + 1); c = idx.getX(fi * 3 + 2)
        } else {
          a = fi * 3; b = fi * 3 + 1; c = fi * 3 + 2
        }
        const fmtV = (i: number) => {
          const u = uvAttr.getX(i).toFixed(4), v = uvAttr.getY(i).toFixed(4)
          const x = positionAttr.getX(i).toFixed(2), y = positionAttr.getY(i).toFixed(2), z = positionAttr.getZ(i).toFixed(2)
          return `v${i}(uv=${u},${v} xyz=${x},${y},${z})`
        }
        const fa = fmtV(a), fb = fmtV(b), fc = fmtV(c)
        const du = Math.max(uvAttr.getX(a), uvAttr.getX(b), uvAttr.getX(c)) - Math.min(uvAttr.getX(a), uvAttr.getX(b), uvAttr.getX(c))
        const dv = Math.max(uvAttr.getY(a), uvAttr.getY(b), uvAttr.getY(c)) - Math.min(uvAttr.getY(a), uvAttr.getY(b), uvAttr.getY(c))
        return `hitFace[${fi}]: ${fa} | ${fb} | ${fc} | faceUvSpan=(dU=${du.toFixed(4)} dV=${dv.toFixed(4)})`
      })(),
      `uvCalibrationTransform: ${meta.uvCalibrationTransform ?? 'n/a'}`,
      `uvCalibrationScale: [${meta.uvCalibrationScaleU?.toFixed(6) ?? 'n/a'}, ${meta.uvCalibrationScaleV?.toFixed(6) ?? 'n/a'}]`,
      `uvCalibrationTargetSource: ${meta.uvCalibrationTargetSource ?? 'n/a'}`,
      (() => {
        // Dump every vertex (position + every available UV channel) and every triangle's indices
        // so we can see whether our decoded UVs match what the mesh actually authored.
        if (!positionAttr) return '\n--- Full Mesh Part Dump ---\nno position attribute'
        const lines: string[] = ['', '--- Full Mesh Part Dump ---']
        const channelNames = ['uv', 'uv1', 'uv2', 'uv3', 'uv4', 'uv5', 'uv6', 'uv7']
        const channels: Array<{ name: string; attr: THREE.BufferAttribute }> = []
        for (const name of channelNames) {
          const a = geometry?.getAttribute(name) as THREE.BufferAttribute | undefined
          if (a && a.itemSize >= 2 && a.count > 0) channels.push({ name, attr: a })
        }
        lines.push(`channels: position(itemSize=${positionAttr.itemSize}, count=${positionAttr.count}) ${channels.map((c) => `${c.name}(itemSize=${c.attr.itemSize}, count=${c.attr.count})`).join(' | ')}`)
        const vertCount = positionAttr.count
        for (let i = 0; i < vertCount; i++) {
          const x = positionAttr.getX(i).toFixed(3)
          const y = positionAttr.getY(i).toFixed(3)
          const z = positionAttr.getZ(i).toFixed(3)
          const uvs = channels.map((c) => `${c.name}=(${c.attr.getX(i).toFixed(4)},${c.attr.getY(i).toFixed(4)})`).join(' ')
          lines.push(`  vtx[${i}] pos=(${x},${y},${z}) ${uvs}`)
        }
        const idx = geometry?.index
        if (idx) {
          const triCount2 = Math.floor(idx.count / 3)
          for (let t = 0; t < triCount2; t++) {
            lines.push(`  tri[${t}] = (${idx.getX(t * 3)}, ${idx.getX(t * 3 + 1)}, ${idx.getX(t * 3 + 2)})`)
          }
        } else {
          lines.push('  (no index buffer)')
        }
        const dd = (geometry?.userData as any)?.decodeDebug
        if (typeof dd === 'string' && dd.length > 0) {
          lines.push('')
          lines.push('--- MSH Decoder Debug ---')
          lines.push(dd)
        }
        return lines.join('\n')
      })(),
      '',
      '--- UV Application Debug ---',
      `uvDebug.called: ${(geometry?.userData?.uvDebug as any)?.called ?? false}`,
      `uvDebug.hasPosition: ${(geometry?.userData?.uvDebug as any)?.hasPosition ?? 'n/a'}`,
      `uvDebug.positionCount: ${(geometry?.userData?.uvDebug as any)?.positionCount ?? 'n/a'}`,
      `uvDebug.hasUvs: ${(geometry?.userData?.uvDebug as any)?.hasUvs ?? 'n/a'}`,
      `uvDebug.uvsLength: ${(geometry?.userData?.uvDebug as any)?.uvsLength ?? 'n/a'}`,
      `uvDebug.requiredUvLength: ${(geometry?.userData?.uvDebug as any)?.requiredUvLength ?? 'n/a'}`,
      `uvDebug.hasUvChannel: ${(geometry?.userData?.uvDebug as any)?.hasUvChannel ?? 'n/a'}`,
      `uvDebug.earlyReturn: ${(geometry?.userData?.uvDebug as any)?.earlyReturn ?? 'n/a'}`,
      `uvDebug.earlyReturnReason: ${(geometry?.userData?.uvDebug as any)?.earlyReturnReason ?? 'n/a'}`,
      `uvDebug.shaderPath: ${(geometry?.userData?.uvDebug as any)?.shaderPath ?? 'n/a'}`,
      `uvDebug.debugKey: ${(geometry?.userData?.uvDebug as any)?.debugKey ?? 'n/a'}`,
      '',
      ...(((geometry?.userData as any)?.shaderDebugChunks) 
        ? [
            '--- Shader IFF Structure ---',
            ...((geometry?.userData as any)?.shaderDebugChunks as string[]),
            '',
          ]
        : meta.shaderDebugChunks && meta.shaderDebugChunks.length > 0
          ? [
              '--- Shader IFF Structure ---',
              ...meta.shaderDebugChunks,
              '',
            ]
          : []),
      ...((() => {
        const effectChunks = meta.effectPath ? getEffectDebugChunks(meta.effectPath) : undefined
        if (!effectChunks || effectChunks.length === 0) return []
        return [
          `--- Effect IFF Structure (${meta.effectPath}) ---`,
          ...effectChunks,
          '',
        ]
      })()),
      `nearOverlapHitCount: ${overlapHits.length}`,
      ...overlapLines,
      '',
      `bindingRow: part ${meta.partIndex ?? 'n/a'}: uv=${meta.primaryUvSetIndex ?? 'n/a'} base=[${formatMode(meta.baseAddressU)},${formatMode(meta.baseAddressV)}] ${meta.baseTexturePath ?? 'n/a'} | normal=[${formatMode(meta.normalAddressU)},${formatMode(meta.normalAddressV)}] ${meta.normalTexturePath ?? 'n/a'} | detailUv=${meta.secondaryUvSetIndex ?? 'n/a'} detail=[${formatMode(meta.detailAddressU)},${formatMode(meta.detailAddressV)}] ${meta.detailTexturePath ?? 'n/a'} | trace=${meta.chunkTrace ?? 'n/a'}`,
    ]

    return lines.join('\n')
  }

  const cacheTexture = (key: string, texture: THREE.Texture) => {
    const cache = textureCacheRef.current
    const order = textureAccessOrderRef.current

    // Estimate texture memory (rough calculation)
    const texelCount = ((texture.image as any)?.width || 512) * ((texture.image as any)?.height || 512)
    const estimatedSize = texelCount * 4 // RGBA

    // Remove oldest textures if cache is too large
    while (textureSizeRef.current + estimatedSize > MAX_TEXTURE_CACHE_SIZE && cache.size > 0) {
      const oldest = order.shift()
      if (oldest && cache.has(oldest)) {
        const oldTexture = cache.get(oldest)!
        textureSizeRef.current -= ((oldTexture.image as any)?.width || 512) * ((oldTexture.image as any)?.height || 512) * 4
        oldTexture.dispose()
        cache.delete(oldest)
      }
    }

    // Add new texture
    cache.set(key, texture)
    order.push(key)
    textureSizeRef.current += estimatedSize
  }

  useEffect(() => {
    const host = containerRef.current;
    if (!host) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, precision: 'highp', powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); // Cap pixel ratio for performance
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.LinearToneMapping;
    renderer.toneMappingExposure = 1;
    renderer.localClippingEnabled = true;
    renderer.shadowMap.enabled = false; // Disable shadows for better performance
    // Install global plane permanently so NUM_CLIPPING_PLANES=1 is baked into shaders at compile time.
    // Never change the array length — only mutate plane.constant — to avoid shader recompilation.
    renderer.clippingPlanes = [globalCutawayPlaneRef.current];
    // 1×1 white placeholder map — shaders compile with USE_MAP=1 from the start.
    const colorData = new Uint8Array([255, 255, 255, 255])
    const colorTex = new THREE.DataTexture(colorData, 1, 1, THREE.RGBAFormat)
    colorTex.colorSpace = THREE.SRGBColorSpace
    colorTex.needsUpdate = true
    placeholderColorTexRef.current = colorTex
    const normalData = new Uint8Array([128, 128, 255, 255])
    const normalTex = new THREE.DataTexture(normalData, 1, 1, THREE.RGBAFormat)
    normalTex.colorSpace = THREE.NoColorSpace
    normalTex.needsUpdate = true
    placeholderNormalTexRef.current = normalTex
    rendererRef.current = renderer;
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0f2026');
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(52, 1, 0.1, 2500);
    camera.position.set(24, 22, 24);
    cameraRef.current = camera;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 4;
    controls.maxDistance = 160;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.target.set(0, 2, 0);
    controlsRef.current = controls;

    textureLoaderRef.current = new THREE.TextureLoader();
    ddsLoaderRef.current = new DDSLoader();
    tgaLoaderRef.current = new TGALoader();

    scene.add(new THREE.AmbientLight('#ffffff', 0.92));
    const keyLight = new THREE.DirectionalLight('#ffffff', 1.05);
    keyLight.position.set(32, 42, 18);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight('#ffffff', 0.5);
    fillLight.position.set(-20, 16, -22);
    scene.add(fillLight);

    const grid = new THREE.GridHelper(180, 90, '#5d8a8b', '#244448');
    scene.add(grid);

    const previewGroup = new THREE.Group();
    scene.add(previewGroup);
    previewGroupRef.current = previewGroup;

    const resize = () => {
      const width = Math.max(host.clientWidth, 100);
      const height = Math.max(host.clientHeight, 100);
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };

    resize();
    window.addEventListener('resize', resize);

    let animationFrame = 0;
    let needsRender = true;

    const animate = () => {
      if (needsRender) {
        controls.update();
        renderer.render(scene, camera);
        needsRender = false;
      }
      animationFrame = window.requestAnimationFrame(animate);
    };
    animate();

    const requestRender = () => {
      needsRender = true;
    };
    requestRenderRef.current = requestRender;

    controls.addEventListener('change', requestRender);
    window.addEventListener('resize', requestRender);

    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()
    let dragStartX = 0
    let dragStartY = 0
    let dragPointerId: number | null = null

    const onPointerDown = (event: PointerEvent) => {
      if (event.button !== 0) return
      dragStartX = event.clientX
      dragStartY = event.clientY
      dragPointerId = event.pointerId
    }

    const onPointerUp = (event: PointerEvent) => {
      if (event.button !== 0) return
      if (dragPointerId !== null && event.pointerId !== dragPointerId) return

      const moveDistance = Math.hypot(event.clientX - dragStartX, event.clientY - dragStartY)
      dragPointerId = null
      if (moveDistance > 5) return

      const canvasRect = renderer.domElement.getBoundingClientRect()
      if (canvasRect.width <= 0 || canvasRect.height <= 0) return

      pointer.x = ((event.clientX - canvasRect.left) / canvasRect.width) * 2 - 1
      pointer.y = -((event.clientY - canvasRect.top) / canvasRect.height) * 2 + 1
      raycaster.setFromCamera(pointer, camera)

      const group = previewGroupRef.current
      if (!group) return

      const intersections = raycaster.intersectObjects(group.children, true)
      const picked = intersections.find((entry) => Boolean((entry.object as THREE.Object3D).userData?.surfacePick))
      if (!picked) return

      const meta = (picked.object.userData.surfacePick as SurfacePickMetadata | undefined)
      if (!meta) return

      setPickedSurfaceDump(buildSurfaceDump(picked, meta, intersections))
      setPickCopyStatus('')
    }

    renderer.domElement.addEventListener('pointerdown', onPointerDown)
    renderer.domElement.addEventListener('pointerup', onPointerUp)

    return () => {
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.domElement.removeEventListener('pointerup', onPointerUp)
      window.cancelAnimationFrame(animationFrame);
      window.removeEventListener('resize', resize);
      controls.dispose();
      textureCacheRef.current.forEach((texture) => texture.dispose());
      textureAccessOrderRef.current = [];
      textureSizeRef.current = 0;
      textureCacheRef.current.clear();
      geometryCacheRef.current.forEach((geom) => geom.dispose());
      geometryCacheRef.current.clear();
      placeholderColorTexRef.current?.dispose()
      placeholderColorTexRef.current = null
      placeholderNormalTexRef.current?.dispose()
      placeholderNormalTexRef.current = null
      if (compileTimerRef.current !== null) {
        clearTimeout(compileTimerRef.current)
        compileTimerRef.current = null
      }
      renderer.dispose();
      host.removeChild(renderer.domElement);
    };
  }, []);

  useEffect(() => {
    const previewGroup = previewGroupRef.current
    const controls = controlsRef.current
    const camera = cameraRef.current
    if (!previewGroup || !controls || !camera) return

    controls.screenSpacePanning = interiorAssist
    controls.minDistance = interiorAssist ? 0.75 : 4
    controls.maxDistance = interiorAssist ? 220 : 160
    controls.maxPolarAngle = interiorAssist ? Math.PI * 0.92 : Math.PI * 0.49
    controls.zoomSpeed = interiorAssist ? 1.35 : 1
    controls.panSpeed = interiorAssist ? 1.2 : 1
    camera.near = interiorAssist ? 0.03 : 0.1
    camera.updateProjectionMatrix()

    while (previewGroup.children.length > 0) {
      const child = previewGroup.children[0]
      previewGroup.remove(child)
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose()
        if (Array.isArray(child.material)) {
          const materials = child.material as THREE.Material[]
          materials.forEach((m) => {
            m.userData = {
              ...m.userData,
              __previewDisposed: true,
            }
            m.dispose()
          })
        }
        else {
          child.material.userData = {
            ...child.material.userData,
            __previewDisposed: true,
          }
          child.material.dispose()
        }
      }
    }
    rootCutawayMaterialsRef.current = []
    allMaterialsRef.current = []

    if (!building) return

    function applyMeshUvs(
      geometry: THREE.BufferGeometry,
      uvs?: number[],
      debugKey?: string,
      hasUvChannel?: boolean,
      shaderPath?: string,
      scaleU?: number,
      scaleV?: number,
    ): void {
      const position = geometry.getAttribute('position')
      
      // Store UV application debug info in geometry for surface pick dump
      const uvDebug = {
        called: true,
        debugKey,
        hasPosition: Boolean(position),
        positionCount: position?.count ?? 0,
        hasUvs: Boolean(uvs),
        uvsLength: uvs?.length ?? 0,
        hasUvChannel,
        shaderPath,
        requiredUvLength: position ? position.count * 2 : 0,
        earlyReturn: false,
        earlyReturnReason: '',
      }
      
      if (!position || position.count === 0 || !uvs || uvs.length < position.count * 2) {
        // Data-only mode: never synthesize planar UVs.
        uvDebug.earlyReturn = true
        uvDebug.earlyReturnReason = !position ? 'no position' : position.count === 0 ? 'zero vertices' : !uvs ? 'no uvs' : 'insufficient uv data'
        geometry.userData.uvDebug = uvDebug
        
        if (debugKey && hasUvChannel !== false && !uvFallbackLoggedRef.current.has(debugKey)) {
          uvFallbackLoggedRef.current.add(debugKey)
          console.warn('[UV] Missing declared UV data; leaving mesh without UV attribute', {
            key: debugKey,
            vertexCount: position?.count ?? 0,
            hasProvidedUvs: Boolean(uvs && uvs.length > 0),
            providedUvCount: uvs?.length ?? 0,
          })
        }
        return
      }
      
      geometry.userData.uvDebug = uvDebug
      
      // Store shader debug info for surface pick dump
      if (shaderPath) {
        const debugChunks = getShaderDebugChunks(shaderPath)
        if (debugChunks) {
          geometry.userData.shaderDebugChunks = debugChunks
        }
      }

      let normalizedUvs = uvs.slice(0, position.count * 2)
      
        // Calculate initial UV range
        let minU = Number.POSITIVE_INFINITY, maxU = Number.NEGATIVE_INFINITY
        let minV = Number.POSITIVE_INFINITY, maxV = Number.NEGATIVE_INFINITY
        for (let i = 0; i + 1 < normalizedUvs.length; i += 2) {
          const u = normalizedUvs[i], v = normalizedUvs[i + 1]
          if (!Number.isFinite(u) || !Number.isFinite(v)) continue
          if (u < minU) minU = u; if (u > maxU) maxU = u
          if (v < minV) minV = v; if (v > maxV) maxV = v
        }
      
      // Apply shader-defined texture coordinate scales from TCSC chunk (if present)
      // Note: Most SWG meshes don't have TCSC chunks - UV scales are baked into the UV coordinates
      if ((scaleU && scaleU !== 1.0) || (scaleV && scaleV !== 1.0)) {
        const uScale = scaleU ?? 1.0
        const vScale = scaleV ?? 1.0
        console.log(`[UV Scale] Applying TCSC shader scales: U=${uScale.toFixed(3)}, V=${vScale.toFixed(3)} to ${debugKey || 'unknown'}`)
        for (let i = 0; i + 1 < normalizedUvs.length; i += 2) {
          normalizedUvs[i] *= uScale
          normalizedUvs[i + 1] *= vScale
        }
      }
      
      // Recalculate UV range for logging
      minU = Number.POSITIVE_INFINITY
      minV = Number.POSITIVE_INFINITY
      maxU = Number.NEGATIVE_INFINITY
      maxV = Number.NEGATIVE_INFINITY
      for (let i = 0; i + 1 < normalizedUvs.length; i += 2) {
        const u = normalizedUvs[i]
        const v = normalizedUvs[i + 1]
        if (Number.isFinite(u) && Number.isFinite(v)) {
          if (u < minU) minU = u
          if (u > maxU) maxU = u
          if (v < minV) minV = v
          if (v > maxV) maxV = v
        }
      }
      
      console.log(`[UV Apply] ${debugKey || 'unknown'}:`)
      console.log(`  Vertex count: ${position.count}, UV pairs: ${normalizedUvs.length / 2}`)
      console.log(`  UV Range: U[${minU.toFixed(3)}, ${maxU.toFixed(3)}], V[${minV.toFixed(3)}, ${maxV.toFixed(3)}]`)
      console.log(`  UV Tiling: ~${Math.abs(maxU - minU).toFixed(1)}x horizontal, ~${Math.abs(maxV - minV).toFixed(1)}x vertical`)
      console.log(`  Shader: ${shaderPath || 'none'}`)
      console.log(`  TCSC scales applied: ${scaleU ? 'U=' + scaleU.toFixed(3) : 'none'}, ${scaleV ? 'V=' + scaleV.toFixed(3) : 'none'}`)
      
      geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(normalizedUvs), 2))
    }

    function pickUvSet(part: { positions: number[]; uvs?: number[]; uvs1?: number[]; uvSets?: number[][] }, uvSetIndex?: number): number[] | undefined {
      const vertexCount = Math.floor(part.positions.length / 3)
      const required = vertexCount * 2
      const valid = (set?: number[]) => Boolean(set && set.length >= required)

      // Deterministic resolution order:
      // 1) explicit requested set index
      // 2) legacy named channel for index 0/1
      // 3) first declared set in deterministic order (uvSets[0], then uvs, then uvs1)
      if (uvSetIndex !== undefined) {
        if (part.uvSets && uvSetIndex >= 0 && uvSetIndex < part.uvSets.length && valid(part.uvSets[uvSetIndex])) {
          return part.uvSets[uvSetIndex]
        }
        if (uvSetIndex === 0 && valid(part.uvs)) return part.uvs
        if (uvSetIndex === 1 && valid(part.uvs1)) return part.uvs1
      }

      if (part.uvSets) {
        for (const set of part.uvSets) {
          if (valid(set)) return set
        }
      }
      if (valid(part.uvs)) return part.uvs
      if (valid(part.uvs1)) return part.uvs1
      return undefined
    }

    function pickUvSetForShader(
      part: { positions: number[]; uvs?: number[]; uvs1?: number[]; uvSets?: number[][] },
      uvSetIndex: number | undefined,
      _shaderPath?: string,
    ): number[] | undefined {
      return pickUvSet(part, uvSetIndex)
    }

    const pendingTextureLoads = new Set<string>()
    const buildEpoch = previewBuildEpochRef.current + 1
    previewBuildEpochRef.current = buildEpoch

    // Expose the current applyTextureToMaterial closure so the renderTextures toggle
    // effect can reapply textures without triggering a full rebuild.
    applyTextureToMaterialRef.current = applyTextureToMaterial
    applyStoredShaderMaterialStateRef.current = applyStoredShaderMaterialState

    const isMaterialUsable = (material: THREE.Material): boolean => {
      if (previewBuildEpochRef.current !== buildEpoch) return false
      return material.userData?.__previewDisposed !== true
    }

    const hydrateCompressedTexture = (texture: THREE.Texture) => {
      const compressed = texture as THREE.CompressedTexture
      if (!compressed.isCompressedTexture) return

      const anyTexture = compressed as unknown as {
        image?: { width?: number; height?: number }
        mipmaps?: Array<{ data?: ArrayBufferView; width?: number; height?: number }>
        source?: {
          data?:
            | { width?: number; height?: number; data?: ArrayBufferView }
            | Array<{ width?: number; height?: number; data?: ArrayBufferView }>
        }
      }

      const sourceData = anyTexture.source?.data
      if (Array.isArray(sourceData) && sourceData.length > 0) {
        if (!anyTexture.mipmaps || anyTexture.mipmaps.length === 0) {
          anyTexture.mipmaps = sourceData
            .filter((entry) => Number.isFinite(entry.width) && Number.isFinite(entry.height) && Boolean(entry.data))
            .map((entry) => ({
              data: entry.data,
              width: entry.width,
              height: entry.height,
            }))
        }
      }

      if (!Array.isArray(anyTexture.mipmaps)) {
        anyTexture.mipmaps = []
      }

      const imageHasDims = Boolean(
        anyTexture.image &&
        Number.isFinite(anyTexture.image.width) &&
        Number.isFinite(anyTexture.image.height),
      )
      if (imageHasDims) return

      const sourceSingle = !Array.isArray(sourceData) ? sourceData : undefined
      const sourceHasDims = Boolean(
        sourceSingle && Number.isFinite(sourceSingle.width) && Number.isFinite(sourceSingle.height),
      )
      if (sourceHasDims) {
        anyTexture.image = {
          width: sourceSingle?.width,
          height: sourceSingle?.height,
        }
      } else {
        const mip = anyTexture.mipmaps?.[0]
        if (mip && Number.isFinite(mip.width) && Number.isFinite(mip.height)) {
          anyTexture.image = {
            width: mip.width,
            height: mip.height,
          }
        }
      }

      // Avoid invalid mip-state uploads when only base level exists/unknown.
      compressed.generateMipmaps = false
      if (!anyTexture.mipmaps || anyTexture.mipmaps.length <= 1) {
        compressed.minFilter = THREE.LinearFilter
      }
    }

    const isTextureRenderable = (texture: THREE.Texture): boolean => {
      if ((texture as THREE.CompressedTexture).isCompressedTexture) {
        hydrateCompressedTexture(texture)
        const compressed = texture as THREE.CompressedTexture
        if (compressed.mipmaps && compressed.mipmaps.length > 0) return true
        const image = compressed.image as { width?: number; height?: number } | undefined
        return Boolean(image && Number.isFinite(image.width) && Number.isFinite(image.height))
      }

      const image = texture.image as { width?: number; height?: number; data?: unknown } | undefined
      if (image && Number.isFinite(image.width) && Number.isFinite(image.height)) return true

      const sourceData = (texture as unknown as { source?: { data?: { width?: number; height?: number; data?: unknown } } }).source?.data
      return Boolean(
        sourceData &&
        Number.isFinite(sourceData.width) &&
        Number.isFinite(sourceData.height),
      )
    }

    const toThreeWrapping = (mode?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce'): THREE.Wrapping => {
      if (mode === 'mirror') return THREE.MirroredRepeatWrapping
      if (mode === 'clamp' || mode === 'border') return THREE.ClampToEdgeWrapping
      if (mode === 'mirroronce') return THREE.MirroredRepeatWrapping
      return THREE.RepeatWrapping
    }

    const configureTexture = (
      texture: THREE.Texture,
      mode: 'color' | 'normal' = 'color',
      texturePath?: string,
      addressU?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce',
      addressV?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce',
      mipmapFilter?: 'none' | 'point' | 'linear' | 'anisotropic' | 'flatcubic' | 'gaussiancubic',
      minificationFilter?: 'none' | 'point' | 'linear' | 'anisotropic' | 'flatcubic' | 'gaussiancubic',
      magnificationFilter?: 'none' | 'point' | 'linear' | 'anisotropic' | 'flatcubic' | 'gaussiancubic',
    ) => {
      const toThreeMinFilter = (
        mip?: 'none' | 'point' | 'linear' | 'anisotropic' | 'flatcubic' | 'gaussiancubic',
        min?: 'none' | 'point' | 'linear' | 'anisotropic' | 'flatcubic' | 'gaussiancubic',
      ): THREE.TextureFilter => {
        const mipMode = mip ?? 'linear'
        const minMode = min ?? 'linear'

        const minIsPoint = minMode === 'point'
        const minIsNone = minMode === 'none'
        const mipIsPoint = mipMode === 'point'
        const mipIsNone = mipMode === 'none'

        if (mipIsNone || minIsNone) {
          return minIsPoint ? THREE.NearestFilter : THREE.LinearFilter
        }

        if (mipIsPoint) {
          return minIsPoint ? THREE.NearestMipMapNearestFilter : THREE.LinearMipMapNearestFilter
        }

        return minIsPoint ? THREE.NearestMipMapLinearFilter : THREE.LinearMipMapLinearFilter
      }

      const toThreeMagFilter = (
        mag?: 'none' | 'point' | 'linear' | 'anisotropic' | 'flatcubic' | 'gaussiancubic',
      ): THREE.MagnificationTextureFilter => {
        if (mag === 'point') return THREE.NearestFilter
        return THREE.LinearFilter
      }

      // DDS/TGA loaders handle orientation correctly - flipY should be false for pre-flipped formats
      // but true for standard formats like PNG/JPG. Let loaders set this automatically.
      // Most DDS/TGA files are already in the correct orientation for OpenGL.
      if (texturePath) {
        const lower = texturePath.toLowerCase()
        // DDS and TGA files are typically pre-flipped for Direct3D/OpenGL
        texture.flipY = !(lower.endsWith('.dds') || lower.endsWith('.tga'))
      }
      
      if (texturePath) {
        texture.userData = {
          ...(texture.userData ?? {}),
          swgTexturePath: texturePath.toLowerCase(),
        }
      }
      texture.colorSpace = mode === 'normal' ? THREE.NoColorSpace : THREE.SRGBColorSpace
      texture.wrapS = toThreeWrapping(addressU)
      texture.wrapT = toThreeWrapping(addressV)
      
      // CRITICAL: Keep texture.repeat at (1,1) so UV coordinates control tiling
      texture.repeat.set(1, 1)
      texture.offset.set(0, 0)
      
      texture.minFilter = toThreeMinFilter(mipmapFilter, minificationFilter)
      texture.magFilter = toThreeMagFilter(magnificationFilter)
      const maxAniso = rendererRef.current?.capabilities.getMaxAnisotropy() ?? 1
      const requestedAniso =
        mipmapFilter === 'anisotropic' ||
        minificationFilter === 'anisotropic' ||
        magnificationFilter === 'anisotropic'
      texture.anisotropy = requestedAniso ? Math.min(maxAniso, 16) : 1
      texture.needsUpdate = true
      
      console.log(`[Texture Config] ${texturePath || 'unknown'}:`)
      console.log(`  Mode: ${mode}, ColorSpace: ${texture.colorSpace === THREE.SRGBColorSpace ? 'SRGB' : 'NoColorSpace'}`)
      console.log(`  Wrap: U=${addressU || 'wrap'}, V=${addressV || 'wrap'}`)
      console.log(`  Filter: Mip=${mipmapFilter || 'linear'}, Min=${minificationFilter || 'linear'}, Mag=${magnificationFilter || 'linear'}`)
      console.log(`  Anisotropy: ${texture.anisotropy}, FlipY: ${texture.flipY}, Repeat: (${texture.repeat.x}, ${texture.repeat.y})`)
    }

    function applyTextureToMaterial(
      templatePath: string | undefined,
      material: THREE.MeshPhongMaterial,
      context: 'object' | 'root',
      _attempted: Set<string> = new Set(),
    ): void {
      if (!renderTexturesRef.current || !templatePath) return

      const altTextureCandidates = (rawPath: string): string[] => {
        const lower = rawPath.toLowerCase()
        if (lower.endsWith('.dds')) return [lower, `${lower.slice(0, -4)}.tga`]
        if (lower.endsWith('.tga')) return [lower, `${lower.slice(0, -4)}.dds`]
        return [lower]
      }

      const directTextureMeta = resolvedTextureAliasRef.current.get(templatePath) ?? resolvedTextures[templatePath]
      let textureMeta = directTextureMeta
      if (!textureMeta) {
        const expectedTexturePath = (material.userData?.expectedTexturePath as string | undefined)
        if (expectedTexturePath) {
          const candidates = altTextureCandidates(expectedTexturePath)
          const candidateFiles = new Set(candidates.map((value) => value.split('/').pop() ?? value))
          textureMeta = Object.values(resolvedTextures).find((entry) => {
            if (!entry?.texturePath) return false
            const entryPath = entry.texturePath.toLowerCase()
            if (candidates.includes(entryPath)) return true
            const entryFile = entryPath.split('/').pop() ?? entryPath
            return candidateFiles.has(entryFile)
          })
          if (textureMeta) {
            // Persist alias so per-part lookups and dumps stop missing after a successful fallback hit.
            resolvedTextureAliasRef.current.set(templatePath, textureMeta)
          }
        }
      }
      if (!textureMeta) {
        return
      }

      const key = textureMeta.url
      if (_attempted.has(key)) return

      const findAlternateTextureMeta = (): ResolvedTextureMap[string] | undefined => {
        if (!textureMeta?.texturePath) return undefined
        if (textureMeta.alternateUrl && textureMeta.alternateTexturePath && textureMeta.alternateUrl !== key) {
          return {
            ...textureMeta,
            url: textureMeta.alternateUrl,
            texturePath: textureMeta.alternateTexturePath,
            alternateUrl: key,
            alternateTexturePath: textureMeta.texturePath,
          }
        }

        const currentPath = textureMeta.texturePath.toLowerCase()
        const alternates = altTextureCandidates(currentPath).filter((value) => value !== currentPath)
        if (alternates.length === 0) return undefined

        const allEntries = Object.values(resolvedTextures)
        for (const altPath of alternates) {
          const direct = allEntries.find((entry) => entry?.texturePath?.toLowerCase() === altPath)
          if (direct && direct.url !== key) return direct
        }

        // Fallback by filename if path prefix differs but basename matches.
        const altFiles = new Set(alternates.map((value) => value.split('/').pop() ?? value))
        return allEntries.find((entry) => {
          if (!entry?.texturePath || entry.url === key) return false
          const file = entry.texturePath.toLowerCase().split('/').pop() ?? entry.texturePath.toLowerCase()
          return altFiles.has(file)
        })
      }

      const cached = textureCacheRef.current.get(key)
      if (cached && isTextureRenderable(cached)) {
        material.map = cached

        const normalKey = textureMeta.normalUrl
        if (normalKey) {
          const normalCached = textureCacheRef.current.get(normalKey)
          if (normalCached && isTextureRenderable(normalCached)) {
            material.normalMap = normalCached
            material.normalScale.set(1, 1)
          } else if (!pendingTextureLoads.has(normalKey) && textureMeta.normalTexturePath) {
            const normalLower = textureMeta.normalTexturePath.toLowerCase()
            const normalLoader = normalLower.endsWith('.dds')
              ? ddsLoaderRef.current
              : normalLower.endsWith('.tga')
                ? tgaLoaderRef.current
                : textureLoaderRef.current
            if (normalLoader) {
              pendingTextureLoads.add(normalKey)
              normalLoader.load(
                normalKey,
                (normalTexture: THREE.Texture) => {
                  pendingTextureLoads.delete(normalKey)
                  if (!isMaterialUsable(material)) {
                    normalTexture.dispose()
                    return
                  }
                  configureTexture(
                    normalTexture,
                    'normal',
                    textureMeta.normalTexturePath,
                    textureMeta.normalTextureAddressU,
                    textureMeta.normalTextureAddressV,
                  )
                  if (!isTextureRenderable(normalTexture)) return
                  cacheTexture(normalKey, normalTexture)
                  material.normalMap = normalTexture
                  material.normalScale.set(1, 1)
                  material.needsUpdate = true
                  requestRenderRef.current()
                },
                undefined,
                () => {
                  pendingTextureLoads.delete(normalKey)
                },
              )
            }
          }
        }

        material.needsUpdate = true
        return
      }
      if (cached && !isTextureRenderable(cached)) {
        textureCacheRef.current.delete(key)
      }

      if (pendingTextureLoads.has(key)) return

      const lower = textureMeta.texturePath.toLowerCase()
      const loader = lower.endsWith('.dds')
        ? ddsLoaderRef.current
        : lower.endsWith('.tga')
          ? tgaLoaderRef.current
          : textureLoaderRef.current
      if (!loader) return

      console.log(`[Texture Load] Loading texture for ${templatePath}:`)
      console.log(`[Texture Load]   Path: ${textureMeta.texturePath}`)
      console.log(`[Texture Load]   URL: ${key}`)
      console.log(`[Texture Load]   Context: ${context}`)
      
      pendingTextureLoads.add(key)
      loader.load(
        key,
        (texture: THREE.Texture) => {
          pendingTextureLoads.delete(key)
          console.log(`[Texture Load] \u2713 Successfully loaded: ${textureMeta.texturePath}`)
          if (!isMaterialUsable(material)) {
            texture.dispose()
            return
          }
          configureTexture(
            texture,
            'color',
            textureMeta.texturePath,
            textureMeta.textureAddressU,
            textureMeta.textureAddressV,
            textureMeta.textureMipmapFilter,
            textureMeta.textureMinificationFilter,
            textureMeta.textureMagnificationFilter,
          )
          if (!isTextureRenderable(texture)) {
            const anyTexture = texture as unknown as {
              source?: { data?: { width?: number; height?: number } }
            }
            const image = texture.image as { width?: number; height?: number } | undefined
            if (!rejectedTexturePathsRef.current.has(textureMeta.texturePath)) {
              rejectedTexturePathsRef.current.add(textureMeta.texturePath)
              console.warn('[Texture] Loaded texture is not renderable', {
                templatePath,
                texturePath: textureMeta.texturePath,
                url: key,
                context,
                isCompressedTexture: (texture as THREE.CompressedTexture).isCompressedTexture === true,
                format: texture.format,
                type: texture.type,
                mipmaps: (texture as THREE.CompressedTexture).mipmaps?.length ?? 0,
                imageWidth: image?.width,
                imageHeight: image?.height,
                sourceWidth: anyTexture.source?.data?.width,
                sourceHeight: anyTexture.source?.data?.height,
              })
            }
            return
          }

            cacheTexture(key, texture)
          material.map = texture
          material.needsUpdate = true
          console.log(`[Texture Apply] Applied color texture to material: ${textureMeta.texturePath}`)

          const applyNormalFromMeta = (normalMetaUrl: string, normalTexturePath: string) => {
            const cachedNormal = textureCacheRef.current.get(normalMetaUrl)
            if (cachedNormal && isTextureRenderable(cachedNormal)) {
              material.normalMap = cachedNormal
              material.normalScale.set(1, 1)
              return
            }

            if (pendingTextureLoads.has(normalMetaUrl)) return

            const normalLower = normalTexturePath.toLowerCase()
            const normalLoader = normalLower.endsWith('.dds')
              ? ddsLoaderRef.current
              : normalLower.endsWith('.tga')
                ? tgaLoaderRef.current
                : textureLoaderRef.current
            if (!normalLoader) return

            pendingTextureLoads.add(normalMetaUrl)
            normalLoader.load(
              normalMetaUrl,
              (normalTexture: THREE.Texture) => {
                pendingTextureLoads.delete(normalMetaUrl)
                if (!isMaterialUsable(material)) {
                  normalTexture.dispose()
                  return
                }
                configureTexture(
                  normalTexture,
                  'normal',
                  normalTexturePath,
                  textureMeta.normalTextureAddressU,
                  textureMeta.normalTextureAddressV,
                )
                if (!isTextureRenderable(normalTexture)) return
                cacheTexture(normalMetaUrl, normalTexture)
                material.normalMap = normalTexture
                material.normalScale.set(1, 1)
                material.needsUpdate = true
                requestRenderRef.current()
              },
              undefined,
              () => {
                pendingTextureLoads.delete(normalMetaUrl)
              },
            )
          }

          if (textureMeta.normalUrl && textureMeta.normalTexturePath) {
            console.log(`[Normal Map] Applying normal map: ${textureMeta.normalTexturePath}`)
            applyNormalFromMeta(textureMeta.normalUrl, textureMeta.normalTexturePath)
          }

          // ATED is a detail stage; use a conservative MODULATE blend (not 2X) for closer SWG parity.
          if (textureMeta.secondaryUrl && textureMeta.secondaryTexturePath && !material.userData.blendInjected) {
            console.log(`[Secondary Texture] Applying secondary/detail texture: ${textureMeta.secondaryTexturePath}`)
            console.log(`[Secondary Texture]   Blend mode: MODULATE (diffuseColor.rgb *= secondary.rgb)`)
            material.userData.blendInjected = true
            const map2Ref: { current: THREE.Texture | null } = { current: null }
            material.userData.map2Ref = map2Ref

            material.customProgramCacheKey = () => 'swg-blend-modulate'
            material.onBeforeCompile = (shader) => {
              shader.uniforms.map2 = { value: map2Ref.current ?? placeholderColorTexRef.current ?? new THREE.Texture() }
              shader.vertexShader = shader.vertexShader
                .replace(
                  '#include <uv_pars_vertex>',
                  'attribute vec2 uv2;\nvarying vec2 vUv2;\n#include <uv_pars_vertex>',
                )
                .replace(
                  '#include <uv_vertex>',
                  '#include <uv_vertex>\nvUv2 = uv2;',
                )
              shader.fragmentShader = shader.fragmentShader
                .replace(
                  '#include <map_pars_fragment>',
                  'uniform sampler2D map2;\nvarying vec2 vUv2;\n#include <map_pars_fragment>',
                )
                .replace(
                  '#include <map_fragment>',
                  '#include <map_fragment>\n{\n  vec4 secondary = texture2D(map2, vUv2);\n  diffuseColor.rgb *= secondary.rgb;\n}\n',
                )
            }

            const secLower = textureMeta.secondaryTexturePath.toLowerCase()
            const secLoader = secLower.endsWith('.dds')
              ? ddsLoaderRef.current
              : secLower.endsWith('.tga')
                ? tgaLoaderRef.current
                : textureLoaderRef.current
            if (secLoader) {
              secLoader.load(
                textureMeta.secondaryUrl,
                (secTex: THREE.Texture) => {
                  if (!isMaterialUsable(material)) { secTex.dispose(); return }
                  configureTexture(
                    secTex,
                    'color',
                    textureMeta.secondaryTexturePath,
                    textureMeta.secondaryTextureAddressU,
                    textureMeta.secondaryTextureAddressV,
                  )
                  if (!isTextureRenderable(secTex)) return
                  cacheTexture(textureMeta.secondaryUrl!, secTex)
                  map2Ref.current = secTex
                  if (material.userData.map2Ref === map2Ref) {
                    const prog = material as THREE.MeshPhongMaterial & { _shader?: { uniforms: { map2?: { value: THREE.Texture } } } }
                    if (prog._shader?.uniforms.map2) {
                      prog._shader.uniforms.map2.value = secTex
                    }
                  }
                  material.needsUpdate = true
                  requestRenderRef.current()
                },
                undefined,
                () => { /* ignore secondary load error */ }
              )
            }
          }

          requestRenderRef.current()
        },
        undefined,
        () => {
          pendingTextureLoads.delete(key)

          const alternate = findAlternateTextureMeta()
          if (alternate && templatePath) {
            resolvedTextureAliasRef.current.set(templatePath, alternate)
            const nextAttempted = new Set(_attempted)
            nextAttempted.add(key)
            console.warn('[Texture] Primary texture load failed, retrying alternate extension', {
              templatePath,
              failedTexturePath: textureMeta.texturePath,
              alternateTexturePath: alternate.texturePath,
              context,
            })
            applyTextureToMaterial(templatePath, material, context, nextAttempted)
            return
          }

          if (!rejectedTexturePathsRef.current.has(textureMeta.texturePath)) {
            rejectedTexturePathsRef.current.add(textureMeta.texturePath)
            console.error(`[Texture Load] \u2717 Failed to load texture for ${templatePath}:`)
            console.error(`[Texture Load]   Path: ${textureMeta.texturePath}`)
            console.error(`[Texture Load]   URL: ${key}`)
            console.error(`[Texture Load]   Context: ${context}`)
            console.warn('[Texture] Failed to load texture', {
              templatePath,
              texturePath: textureMeta.texturePath,
              url: key,
              context,
            })
          }
        },
      )
    }

    function applyShaderRenderSettings(
      material: THREE.MeshPhongMaterial,
      shaderPath?: string,
    ): void {
      if (!shaderPath) return
      const props = getShaderRenderProps(shaderPath)
      if (!props) return

      const hasPunchout = props.effectTags.some((tag) => tag === 'PNCH' || tag === 'PUNCHOUT')
      if (props.alphaTest || hasPunchout) {
        const alphaRef = props.alphaReference > 0 ? props.alphaReference : (hasPunchout ? 128 : 0)
        material.alphaTest = THREE.MathUtils.clamp(alphaRef / 255, 0, 1)
      } else {
        material.alphaTest = 0
      }

      if (props.transparent || props.alphaBlend) {
        material.transparent = true
        material.depthWrite = false
      } else {
        material.depthWrite = true
      }

      const hasAdditive = props.effectTags.some((tag) => tag === 'ADDT' || tag === 'ADDITIVE' || tag === 'ADD')
      if (hasAdditive) {
        material.blending = THREE.AdditiveBlending
        material.transparent = true
        material.depthWrite = false
      } else {
        material.blending = THREE.NormalBlending
      }

      material.userData = {
        ...material.userData,
        shaderAlphaTest: props.alphaTest,
        shaderAlphaReference: props.alphaReference,
        shaderTransparent: props.transparent,
        shaderAlphaBlend: props.alphaBlend,
        shaderEffectTags: props.effectTags,
      }
    }

    function applyStoredShaderMaterialState(material: THREE.MeshPhongMaterial): void {
      const alphaTest = Boolean(material.userData?.shaderAlphaTest)
      const alphaRef = Number(material.userData?.shaderAlphaReference ?? 0)
      const effectTags = Array.isArray(material.userData?.shaderEffectTags)
        ? (material.userData.shaderEffectTags as string[])
        : []
      const hasPunchout = effectTags.some((tag) => tag === 'PNCH' || tag === 'PUNCHOUT')
      if (alphaTest || hasPunchout) {
        const ref = alphaRef > 0 ? alphaRef : (hasPunchout ? 128 : 0)
        material.alphaTest = THREE.MathUtils.clamp(ref / 255, 0, 1)
      } else {
        material.alphaTest = 0
      }

      const isTransparent = Boolean(material.userData?.shaderTransparent) || Boolean(material.userData?.shaderAlphaBlend)
      const hasAdditive = effectTags.some((tag) => tag === 'ADDT' || tag === 'ADDITIVE' || tag === 'ADD')
      material.transparent = isTransparent || hasAdditive
      material.blending = hasAdditive ? THREE.AdditiveBlending : THREE.NormalBlending
      material.depthWrite = !(material.transparent)
      material.opacity = 1
    }

    const footprintGeometry = new THREE.BoxGeometry(building.width, 0.28, building.depth)
    const footprintMaterial = new THREE.MeshPhongMaterial({
      color: '#2c555a',
      transparent: true,
      opacity: interiorAssist ? 0.18 : 0.75,
      specular: '#000000',
      shininess: 0,
    })
    const footprintMesh = new THREE.Mesh(footprintGeometry, footprintMaterial)
    footprintMesh.position.set(0, 0.14, 0)
    previewGroup.add(footprintMesh)

    const rootVisual = building.rootTemplatePath
      ? resolvedVisuals[normalizeSwgPath(building.rootTemplatePath)]
      : undefined
    if (rootVisual?.mesh && rootVisual.mesh.positions.length >= 9) {
      const rootTemplateKey = building.rootTemplatePath
        ? normalizeSwgPath(building.rootTemplatePath)
        : undefined

      const rootBoundsGeometry = new THREE.BufferGeometry()
      rootBoundsGeometry.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array(rootVisual.mesh.positions), 3),
      )
      if (rootVisual.mesh.indices.length >= 3) {
        rootBoundsGeometry.setIndex(rootVisual.mesh.indices)
      }
      rootBoundsGeometry.computeBoundingBox()
      const rootBounds = rootBoundsGeometry.boundingBox

      if (rootBounds) {
        const center = rootBounds.getCenter(new THREE.Vector3())
        const size = rootBounds.getSize(new THREE.Vector3())
        const widthScale = size.x > 0.001 ? building.width / size.x : 1
        const depthScale = size.z > 0.001 ? building.depth / size.z : 1
        const fitScale = Math.min(widthScale, depthScale)
        const normalizedScale = clamp(fitScale, 0.1, 50)

        const buildingHeight = Math.max(0.1, size.y * normalizedScale)
        resolvedBuildingHeightRef.current = buildingHeight

        const rootPartMeshes = rootVisual.meshParts && rootVisual.meshParts.length > 0
          ? rootVisual.meshParts
          : [rootVisual.mesh]

        for (let partIndex = 0; partIndex < rootPartMeshes.length; partIndex += 1) {
          const part = rootPartMeshes[partIndex]
          if (!part || part.positions.length < 9 || part.indices.length < 3) continue

          const partTextureKey = rootTemplateKey
            ? buildPartTextureKey(rootTemplateKey, partIndex)
            : undefined

          const partGeometry = new THREE.BufferGeometry()
          partGeometry.setAttribute(
            'position',
            new THREE.BufferAttribute(new Float32Array(part.positions), 3),
          )
          partGeometry.setIndex(part.indices)
          const primaryUvSetIndex = rootVisual.meshPartPrimaryUvSetIndices?.[partIndex] ?? 0
          const shaderPath = rootVisual.meshPartShaderPaths?.[partIndex]
          const partDomain = rootVisual.meshPartDomains?.[partIndex] ?? 'exterior'
          const primaryUvs = pickUvSetForShader(
            part,
            primaryUvSetIndex,
            rootVisual.meshPartShaderPaths?.[partIndex],
          )
          const primaryScaleU = rootVisual.meshPartPrimaryScaleU?.[partIndex]
          const primaryScaleV = rootVisual.meshPartPrimaryScaleV?.[partIndex]
          applyMeshUvs(
            partGeometry,
            primaryUvs,
            partTextureKey ?? rootTemplateKey,
            part.hasUvChannel,
            rootVisual.meshPartShaderPaths?.[partIndex],
            primaryScaleU,
            primaryScaleV,
          )

          // Assign secondary UVs (uv2) using TCSS slot mapping when available.
          const secondaryUvSetIndex = rootVisual.meshPartSecondaryUvSetIndices?.[partIndex] ?? 0
          const secondaryUvs = pickUvSetForShader(part, secondaryUvSetIndex, rootVisual.meshPartShaderPaths?.[partIndex])
          if (secondaryUvs && secondaryUvs.length >= Math.floor(part.positions.length / 3) * 2) {
            partGeometry.setAttribute('uv2', new THREE.BufferAttribute(
              new Float32Array(secondaryUvs.slice(0, Math.floor(part.positions.length / 3) * 2)),
              2,
            ))
          }
          // DEBUG: expose every authored UV set as uv3/uv4/... so the surface dump can show them.
          if (part.uvSets && part.uvSets.length > 0) {
            const vc = Math.floor(part.positions.length / 3)
            for (let setIdx = 0; setIdx < part.uvSets.length; setIdx += 1) {
              const src = part.uvSets[setIdx]
              if (!src || src.length < vc * 2) continue
              const attrName = `uv${setIdx + 3}` // uv3 = uvSets[0], uv4 = uvSets[1], etc.
              partGeometry.setAttribute(
                attrName,
                new THREE.BufferAttribute(new Float32Array(src.slice(0, vc * 2)), 2),
              )
            }
          }
          if (part.normals && part.normals.length === part.positions.length) {
            partGeometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(part.normals), 3))
          } else {
            partGeometry.computeVertexNormals()
          }
          partGeometry.translate(-center.x, -rootBounds.min.y, -center.z)
          // DEBUG: stash the MSH decoder summary on the geometry so the surface dump can show it.
          if (part.decodeDebug) {
            partGeometry.userData = { ...partGeometry.userData, decodeDebug: part.decodeDebug }
          }

          const partMaterial = new THREE.MeshPhongMaterial({
            color: '#ffffff',
            map: placeholderColorTexRef.current,
            normalMap: placeholderNormalTexRef.current,
            specular: '#000000',
            shininess: 0,
            side: THREE.DoubleSide,
            transparent: interiorAssist && !renderTexturesRef.current,
            opacity: interiorAssist && !renderTexturesRef.current ? interiorOpacityRef.current : 1,
            depthWrite: true,
          })
          applyShaderRenderSettings(partMaterial, shaderPath)

          if (partDomain === 'interior') {
            // Keep interior overlays from competing with exterior shell surfaces.
            partMaterial.polygonOffset = true
            partMaterial.polygonOffsetFactor = 1
            partMaterial.polygonOffsetUnits = 1
          }

          partMaterial.userData = {
            ...partMaterial.userData,
            expectedTexturePath: rootVisual.meshPartTexturePaths?.[partIndex],
            expectedNormalTexturePath: rootVisual.meshPartNormalTexturePaths?.[partIndex],
            expectedSecondaryTexturePath: rootVisual.meshPartSecondaryTexturePaths?.[partIndex],
          }

          const hasPartTextureBinding = Boolean(
            rootVisual.meshPartTexturePaths && rootVisual.meshPartTexturePaths[partIndex],
          )
          if (hasPartTextureBinding) {
            applyTextureToMaterial(partTextureKey, partMaterial, 'root')
          } else {
            applyTextureToMaterial(rootTemplateKey, partMaterial, 'root')
          }
          allMaterialsRef.current.push({
            material: partMaterial,
            templatePath: hasPartTextureBinding ? partTextureKey : rootTemplateKey,
            context: 'root',
          })

          const partMesh = new THREE.Mesh(partGeometry, partMaterial)
          partMesh.userData = {
            ...partMesh.userData,
            surfacePick: {
              scope: 'root',
              buildingName: building.name,
              templatePath: rootTemplateKey,
              meshPath: rootVisual.meshPath,
              appearancePath: rootVisual.appearancePath,
              shaderPath: rootVisual.meshPartShaderPaths?.[partIndex] ?? rootVisual.shaderPath,
              effectPath: rootVisual.meshPartEffectPaths?.[partIndex] ?? rootVisual.effectPath,
              sourceLabel: rootVisual.sourceLabel,
              textureLookupKey: (rootVisual.meshPartTexturePaths && rootVisual.meshPartTexturePaths[partIndex])
                ? partTextureKey
                : rootTemplateKey,
              partDomain,
              partIndex,
              primaryUvSetIndex: rootVisual.meshPartPrimaryUvSetIndices?.[partIndex],
              secondaryUvSetIndex: rootVisual.meshPartSecondaryUvSetIndices?.[partIndex],
              baseTexturePath: rootVisual.meshPartTexturePaths?.[partIndex] ?? rootVisual.texturePath,
              normalTexturePath: rootVisual.meshPartNormalTexturePaths?.[partIndex] ?? rootVisual.normalTexturePath,
              detailTexturePath: rootVisual.meshPartSecondaryTexturePaths?.[partIndex],
              baseAddressU: rootVisual.meshPartTextureAddressU?.[partIndex],
              baseAddressV: rootVisual.meshPartTextureAddressV?.[partIndex],
              normalAddressU: rootVisual.meshPartNormalTextureAddressU?.[partIndex],
              normalAddressV: rootVisual.meshPartNormalTextureAddressV?.[partIndex],
              detailAddressU: rootVisual.meshPartSecondaryTextureAddressU?.[partIndex],
              detailAddressV: rootVisual.meshPartSecondaryTextureAddressV?.[partIndex],
              chunkTrace: rootVisual.meshPartChunkTrace?.[partIndex],
              uvCalibrationTransform: 'identity',
              uvCalibrationScaleU: 1,
              uvCalibrationScaleV: 1,
              uvCalibrationTargetSource: 'deterministic',
            } satisfies SurfacePickMetadata,
          }
          partMesh.renderOrder = partDomain === 'interior' ? 0 : 1
          partMesh.scale.setScalar(normalizedScale)
          previewGroup.add(partMesh)
        }

        const focusHeight = interiorAssist
          ? clamp(size.y * normalizedScale * 0.38, 1.2, Math.max(2, building.height))
          : Math.max(2, building.height) * 0.32
        controls.target.set(0, focusHeight, 0)
      }

      rootBoundsGeometry.dispose()
    }

    if (!interiorAssist) {
      const boundsGeometry = new THREE.BoxGeometry(building.width, Math.max(2, building.height), building.depth)
      const boundsEdges = new THREE.EdgesGeometry(boundsGeometry)
      const boundsLines = new THREE.LineSegments(
        boundsEdges,
        new THREE.LineBasicMaterial({ color: '#b6f2d9' }),
      )
      boundsLines.position.set(0, Math.max(2, building.height) / 2, 0)
      previewGroup.add(boundsLines)
    }

    building.objects.forEach((object) => {
      const selected = object.id === selectedObjectId
      const resolved = object.templatePath ? resolvedVisuals[normalizeSwgPath(object.templatePath)] : undefined

      let mesh: THREE.Mesh
      if (resolved?.mesh && resolved.mesh.positions.length >= 9) {
        const geometry = new THREE.BufferGeometry()
        const positions = new Float32Array(resolved.mesh.positions)
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))

        if (resolved.mesh.indices.length >= 3) {
          geometry.setIndex(resolved.mesh.indices)
        }

        const objectPrimaryUvSetIndex = resolved.meshPartPrimaryUvSetIndices?.[0] ?? 0
        const objectPrimaryUvs = pickUvSetForShader(
          resolved.mesh,
          objectPrimaryUvSetIndex,
          resolved.meshPartShaderPaths?.[0] ?? resolved.shaderPath,
        )
        applyMeshUvs(
          geometry,
          objectPrimaryUvs,
          object.templatePath ? normalizeSwgPath(object.templatePath) : undefined,
          resolved.mesh.hasUvChannel,
          resolved.meshPartShaderPaths?.[0] ?? resolved.shaderPath,
          resolved.meshPartPrimaryScaleU?.[0],
          resolved.meshPartPrimaryScaleV?.[0],
        )

        geometry.computeBoundingBox()
        geometry.computeBoundingSphere()
        if (resolved.mesh.normals && resolved.mesh.normals.length === resolved.mesh.positions.length) {
          geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(resolved.mesh.normals), 3))
        } else {
          geometry.computeVertexNormals()
        }

        const material = new THREE.MeshPhongMaterial({
          color: selected ? '#f4f5a0' : '#ffffff',
          map: placeholderColorTexRef.current,
          normalMap: placeholderNormalTexRef.current,
          specular: '#000000',
          shininess: 0,
          transparent: false,
          opacity: 1,
          depthWrite: true,
        })
        applyShaderRenderSettings(material, resolved.meshPartShaderPaths?.[0] ?? resolved.shaderPath)
        applyTextureToMaterial(
          object.templatePath ? normalizeSwgPath(object.templatePath) : undefined,
          material,
          'object',
        )
        allMaterialsRef.current.push({
          material,
          templatePath: object.templatePath ? normalizeSwgPath(object.templatePath) : undefined,
          context: 'object',
        })
        mesh = new THREE.Mesh(geometry, material)
        mesh.userData = {
          ...mesh.userData,
          surfacePick: {
            scope: 'object',
            buildingName: building.name,
            objectId: object.id,
            objectName: object.name,
            templatePath: object.templatePath ? normalizeSwgPath(object.templatePath) : undefined,
            meshPath: resolved.meshPath,
            appearancePath: resolved.appearancePath,
            shaderPath: resolved.shaderPath,
            effectPath: resolved.meshPartEffectPaths?.[0] ?? resolved.effectPath,
            sourceLabel: resolved.sourceLabel,
            textureLookupKey: object.templatePath ? normalizeSwgPath(object.templatePath) : undefined,
            partIndex: 0,
            primaryUvSetIndex: resolved.meshPartPrimaryUvSetIndices?.[0],
            secondaryUvSetIndex: resolved.meshPartSecondaryUvSetIndices?.[0],
            baseTexturePath: resolved.meshPartTexturePaths?.[0] ?? resolved.texturePath,
            normalTexturePath: resolved.meshPartNormalTexturePaths?.[0] ?? resolved.normalTexturePath,
            detailTexturePath: resolved.meshPartSecondaryTexturePaths?.[0],
            baseAddressU: resolved.meshPartTextureAddressU?.[0] ?? resolved.textureAddressU,
            baseAddressV: resolved.meshPartTextureAddressV?.[0] ?? resolved.textureAddressV,
            normalAddressU: resolved.meshPartNormalTextureAddressU?.[0] ?? resolved.normalTextureAddressU,
            normalAddressV: resolved.meshPartNormalTextureAddressV?.[0] ?? resolved.normalTextureAddressV,
            detailAddressU: resolved.meshPartSecondaryTextureAddressU?.[0],
            detailAddressV: resolved.meshPartSecondaryTextureAddressV?.[0],
            chunkTrace: resolved.meshPartChunkTrace?.[0],
          } satisfies SurfacePickMetadata,
        }

        const bounds = geometry.boundingBox
        if (bounds) {
          const center = bounds.getCenter(new THREE.Vector3())
          const size = bounds.getSize(new THREE.Vector3())
          const maxAxis = Math.max(size.x, size.y, size.z, 0.001)
          const scale = (1.6 * object.scale) / maxAxis
          mesh.scale.setScalar(scale)
          mesh.position.set(object.x, object.y + (size.y * scale) / 2, object.z)
          mesh.geometry.translate(-center.x, -bounds.min.y, -center.z)
        } else {
          mesh.position.set(object.x, object.y, object.z)
        }
      } else {
        if (!showPlaceholderObjects) return
        const geometry = new THREE.BoxGeometry(1.2 * object.scale, 1.8 * object.scale, 1.2 * object.scale)
        const material = new THREE.MeshPhongMaterial({
          color: selected ? '#f4f5a0' : '#89d8c7',
          map: placeholderColorTexRef.current,
          normalMap: placeholderNormalTexRef.current,
          specular: '#000000',
          shininess: 0,
        })
        applyTextureToMaterial(
          object.templatePath ? normalizeSwgPath(object.templatePath) : undefined,
          material,
          'object',
        )
        allMaterialsRef.current.push({
          material,
          templatePath: object.templatePath ? normalizeSwgPath(object.templatePath) : undefined,
          context: 'object',
        })
        mesh = new THREE.Mesh(geometry, material)
        mesh.position.set(object.x, object.y + 0.9 * object.scale, object.z)
      }

      mesh.rotation.y = (object.yaw * Math.PI) / 180
      previewGroup.add(mesh)
    })

    if (!interiorAssist) {
      controls.target.set(0, Math.max(2, building.height) * 0.32, 0)
    }
    controls.update()
    requestRenderRef.current()

    // Debounce shader pre-compilation so it only starts once deps settle.
    // resolvedTextures / resolvedVisuals update per-texture and would otherwise
    // restart compileAsync on every individual texture load.
    if (compileTimerRef.current !== null) clearTimeout(compileTimerRef.current)
    const epochAtSchedule = buildEpoch
    compileTimerRef.current = setTimeout(() => {
      compileTimerRef.current = null
      if (previewBuildEpochRef.current !== epochAtSchedule) return
      const renderer = rendererRef.current
      const scene = sceneRef.current
      if (!renderer || !scene) return
      setCompilingShaders(true)
      renderer.compileAsync(scene, camera).then(() => {
        setCompilingShaders(false)
      }).catch(() => {
        setCompilingShaders(false)
      })
    }, 500)

    return () => {
      if (compileTimerRef.current !== null) {
        clearTimeout(compileTimerRef.current)
        compileTimerRef.current = null
      }
      if (previewBuildEpochRef.current === buildEpoch) {
        previewBuildEpochRef.current = buildEpoch + 1
      }
    }
  }, [building, interiorAssist, selectedObjectId, resolvedVisuals, resolvedTextures, showPlaceholderObjects])

  // Lightweight effect: toggling renderTextures only swaps material.map on existing materials.
  // No geometry is recreated, so there is no freeze.
  useEffect(() => {
    const entries = allMaterialsRef.current
    if (!entries.length) return
    const placeholder = placeholderColorTexRef.current
    for (const { material, templatePath, context } of entries) {
      if (material.userData.__previewDisposed) continue
      if (!renderTextures) {
        material.map = placeholder
        material.normalMap = placeholderNormalTexRef.current
        material.transparent = interiorAssist
        material.opacity = interiorAssist ? interiorOpacity : 1
        material.depthWrite = true
        material.needsUpdate = true
      } else {
        // Reset to placeholder first so applyTextureToMaterial can load from cache or re-fetch.
        material.map = placeholder
        material.normalMap = placeholderNormalTexRef.current
        applyStoredShaderMaterialStateRef.current(material)
        applyTextureToMaterialRef.current(templatePath, material, context)
      }
    }
    requestRenderRef.current()
  }, [renderTextures, interiorAssist, interiorOpacity])

  // Update global clipping plane constant only — never reassign renderer.clippingPlanes array.
  // Changing the array length would alter NUM_CLIPPING_PLANES and force shader recompilation on every material.
  useEffect(() => {
    const plane = globalCutawayPlaneRef.current

    if (!roofCutaway) {
      // Move plane far above everything — clips nothing, zero GPU cost
      plane.constant = 999999
    } else {
      const bh = resolvedBuildingHeightRef.current
      const minCut = Math.max(0.05, bh * 0.03)
      const maxCut = bh * 1.05
      const cutHeight = minCut + (maxCut - minCut) * clamp(roofCutawayLevel, 0, 1)
      plane.constant = cutHeight
    }

    requestRenderRef.current()
  }, [roofCutaway, roofCutawayLevel])

  return (
    <div className="preview-canvas-wrap">
      <div className="preview-canvas" ref={containerRef} />
      {compilingShaders && (
        <div className="preview-compile-overlay">
          <span className="preview-compile-label">Compiling shaders…</span>
          <div className="preview-compile-bar">
            <div className="preview-compile-bar-fill" />
          </div>
        </div>
      )}
      <div className="preview-pick-panel">
        <div className="preview-pick-header">
          <h3>Clicked Surface Dump</h3>
          <button
            className="btn"
            type="button"
            onClick={() => {
              if (!pickedSurfaceDump) return
              void navigator.clipboard.writeText(pickedSurfaceDump)
                .then(() => setPickCopyStatus('Copied dump to clipboard.'))
                .catch(() => setPickCopyStatus('Copy failed. You can still select and copy below.'))
            }}
          >
            Copy Dump
          </button>
        </div>
        {pickCopyStatus && <p className="muted">{pickCopyStatus}</p>}
        <pre className="preview-pick-dump">{pickedSurfaceDump}</pre>
      </div>
    </div>
  )
})

function FileTreeBrowser({
  node,
  expandedFolders,
  setExpandedFolders,
  onFileClick,
  getTreSource,
}: {
  node: FileTreeNode
  expandedFolders: Set<string>
  setExpandedFolders: (next: Set<string>) => void
  onFileClick: (node: FileTreeNode) => void
  getTreSource: (path: string) => string | undefined
}) {
  if (!node.children) return null

  const sortedChildren = [...node.children].sort((a, b) => {
    if (a.kind !== b.kind) return a.kind === 'folder' ? -1 : 1
    return a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
  })

  return (
    <ul className="file-tree-list">
      {sortedChildren.map((child) => {
        if (child.kind === 'folder') {
          const expanded = expandedFolders.has(child.path)
          return (
            <li key={child.path} className="file-tree-folder">
              <button
                type="button"
                className="folder-label"
                onClick={() => {
                  const next = new Set(expandedFolders)
                  if (expanded) next.delete(child.path)
                  else next.add(child.path)
                  setExpandedFolders(next)
                }}
              >
                {expanded ? 'v' : '>'} {child.name}
              </button>
              {expanded && (
                <FileTreeBrowser
                  node={child}
                  expandedFolders={expandedFolders}
                  setExpandedFolders={setExpandedFolders}
                  onFileClick={onFileClick}
                  getTreSource={getTreSource}
                />
              )}
            </li>
          )
        }

        const treSource = getTreSource(child.path)

        return (
          <li key={child.path} className="file-tree-file">
            <button
              type="button"
              className="file-label"
              title={treSource ? `${child.path} [${treSource}]` : child.path}
              onClick={() => onFileClick(child)}
            >
              <span className="file-label-name">{child.name}</span>
              {treSource && <span className="tre-source-tag">{treSource}</span>}
            </button>
          </li>
        )
      })}
    </ul>
  )
}

function App() {
  const [assets, setAssets] = useState<AssetRecord[]>([])
  const [buildings, setBuildings] = useState<BuildingDocument[]>([])
  const [indexedFiles, setIndexedFiles] = useState<IndexedFileEntry[]>([])
  const [treIndexedEntries, setTreIndexedEntries] = useState<TreIndexedEntry[]>([])

  const [activeBuildingId, setActiveBuildingId] = useState<string | null>(null)
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null)
  const [selectedObjectAssetId, setSelectedObjectAssetId] = useState<string | null>(null)
  const [assetSearch, setAssetSearch] = useState('')
  const [indexingStatus, setIndexingStatus] = useState<string | null>(null)
  const [log, setLog] = useState<string[]>(['Editor initialized.'])
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set())
  const [resolvedVisuals, setResolvedVisuals] = useState<ResolvedVisualMap>({})
  const [resolvedTextures, setResolvedTextures] = useState<ResolvedTextureMap>({})
  const [rootMeshOverridePath, setRootMeshOverridePath] = useState('')
  const [interiorAssist, setInteriorAssist] = useState(true)
  const [interiorOpacity, setInteriorOpacity] = useState(0.22)
  const [roofCutaway, setRoofCutaway] = useState(true)
  const [roofCutawayLevel, setRoofCutawayLevel] = useState(0.6)
  const [showPlaceholderObjects, setShowPlaceholderObjects] = useState(false)
  const [renderTextures, setRenderTextures] = useState(false)

  const looseFileByPathRef = useRef<Map<string, File>>(new Map())
  const treExtractedByPathRef = useRef<Map<string, RepositorySourceFile>>(new Map())
  const treLookupByPathRef = useRef<Map<string, TreLookupRecord>>(new Map())
  const treLookupBuiltRef = useRef<boolean>(false)
  const objectUrlsRef = useRef<Map<string, string>>(new Map())

  const activeBuilding = useMemo(
    () => buildings.find((building) => building.id === activeBuildingId),
    [buildings, activeBuildingId],
  )

  const objectAssets = useMemo(
    () => assets.filter((asset) => asset.kind === 'object'),
    [assets],
  )

  const appearanceAssets = useMemo(
    () => assets.filter((asset) => asset.kind === 'appearance'),
    [assets],
  )

  const filteredAssets = useMemo(() => {
    const term = assetSearch.trim().toLowerCase()
    if (!term) return assets
    return assets.filter((asset) => {
      const haystack = `${asset.name} ${asset.relativePath}`.toLowerCase()
      return haystack.includes(term)
    })
  }, [assets, assetSearch])

  const filteredObjectAssets = useMemo(
    () => filteredAssets.filter((asset) => asset.kind === 'object'),
    [filteredAssets],
  )

  const resolvedTreEntries = useMemo(() => {
    const sorted = [...treIndexedEntries].sort((a, b) => {
      const prio = treArchivePriority(a.treName) - treArchivePriority(b.treName)
      if (prio !== 0) return prio
      return a.treName.localeCompare(b.treName)
    })

    // SIE-style combined repository semantics: later/higher priority archives override earlier ones.
    const byPath = new Map<string, TreIndexedEntry>()
    for (const entry of sorted) {
      byPath.set(entry.path, entry)
    }
    return Array.from(byPath.values())
  }, [treIndexedEntries])

  const resolvedTreByPath = useMemo(() => {
    const byPath = new Map<string, TreIndexedEntry>()
    for (const entry of resolvedTreEntries) {
      byPath.set(entry.path, entry)
    }
    return byPath
  }, [resolvedTreEntries])

  const indexedByPath = useMemo(() => {
    const byPath = new Map<string, IndexedFileEntry>()
    for (const entry of indexedFiles) {
      byPath.set(normalizeSwgPath(entry.relativePath), entry)
    }
    return byPath
  }, [indexedFiles])

  const rootMeshOverrideOptions = useMemo(() => {
    if (!activeBuilding?.rootTemplatePath) return [] as string[]
    const rootKey = normalizeSwgPath(activeBuilding.rootTemplatePath)
    const rootVisual = resolvedVisuals[rootKey]
    const appearancePath = rootVisual?.appearancePath
    if (!appearancePath) return [] as string[]

    const appearanceName = appearancePath.split('/').pop() ?? appearancePath
    const appearanceStem = appearanceName.replace(/\.[^.]+$/, '')
    const stemBase = appearanceStem.replace(/_r\d+(?:_mesh)?$/, '')
    const hints = Array.from(new Set([appearanceStem, stemBase].map((v) => v.toLowerCase())))

    const allPaths = new Set<string>()
    for (const key of indexedByPath.keys()) allPaths.add(key)
    for (const key of resolvedTreByPath.keys()) allPaths.add(key)

    return Array.from(allPaths)
      .filter((path) => path.startsWith('appearance/'))
      .filter((path) => ['.msh', '.lod', '.pob'].includes(extensionOf(path)))
      .filter((path) => hints.some((hint) => path.includes(hint)))
      .sort((a, b) => a.localeCompare(b))
      .slice(0, 300)
  }, [activeBuilding, indexedByPath, resolvedTreByPath, resolvedVisuals])

  const treeData = useMemo(() => {
    const term = assetSearch.trim().toLowerCase()
    const byTreePath = new Map<string, FileTreeItem>()

    for (const asset of assets) {
      const treePath = asset.relativePath
      if (term && !treePath.toLowerCase().includes(term)) continue
      byTreePath.set(treePath, { treePath, source: 'loaded', assetId: asset.id })
    }

    for (const indexed of indexedFiles) {
      const kind = inferAssetKind(indexed.file.name, indexed.relativePath)
      if (!kind) continue
      if (term && !indexed.relativePath.toLowerCase().includes(term)) continue
      if (!byTreePath.has(indexed.relativePath)) {
        byTreePath.set(indexed.relativePath, {
          treePath: indexed.relativePath,
          source: 'indexed',
          relativePath: indexed.relativePath,
        })
      }
    }

    for (const entry of resolvedTreEntries) {
      const treePath = `repository/${entry.path}`
      if (term && !treePath.toLowerCase().includes(term)) continue
      if (!byTreePath.has(treePath)) {
        byTreePath.set(treePath, { treePath, source: 'tre', treEntryId: entry.id })
      }
    }

    const treeInput = Array.from(byTreePath.values()).map((item) => ({
      file: { name: item.treePath.split('/').pop() ?? item.treePath } as File,
      relativePath: item.treePath,
    }))

    return {
      map: byTreePath,
      root: buildFileTree(treeInput),
    }
  }, [assets, indexedFiles, resolvedTreEntries, assetSearch])

  const treSourceByTreePath = useMemo(() => {
    const byPath = new Map<string, string>()
    for (const entry of resolvedTreEntries) {
      byPath.set(`repository/${entry.path}`, entry.treName)
    }
    return byPath
  }, [resolvedTreEntries])

  const lookupRepositoryFile = useCallback(async (path: string): Promise<RepositorySourceFile | null> => {
    const normalized = normalizeSwgPath(path)
    const isTexture = normalized.endsWith('.dds') || normalized.endsWith('.tga')

    // For textures, ONLY use TRE files to ensure exact SWG rendering
    if (!isTexture) {
      const cachedLoose = looseFileByPathRef.current.get(normalized)
      if (cachedLoose) {
        return {
          file: cachedLoose,
          sourceLabel: 'loose/indexed',
        }
      }

      const indexed = indexedByPath.get(normalized)
      if (indexed) {
        looseFileByPathRef.current.set(normalized, indexed.file)
        return {
          file: indexed.file,
          sourceLabel: 'loose/indexed',
        }
      }
    }

    const cachedTre = treExtractedByPathRef.current.get(normalized)
    if (cachedTre) {
      if (isTexture) console.log(`[File Lookup] ✓ Found in cached TRE: ${normalized}`)
      return cachedTre
    }

    const treEntry = resolvedTreByPath.get(normalized)
    let recordHit: TreLookupRecord | null = null
    if (treEntry) {
      if (isTexture) console.log(`[File Lookup] ✓ Found in resolvedTreByPath: ${normalized}`)
      recordHit = {
        treName: treEntry.treName,
        treFile: treEntry.treFile,
        record: treEntry.record,
      }
    } else {
      const cachedLookup = treLookupByPathRef.current.get(normalized)
      if (cachedLookup) {
        if (isTexture) console.log(`[File Lookup] ✓ Found in TRE lookup cache: ${normalized}`)
        recordHit = cachedLookup
      } else {
        if (!treLookupBuiltRef.current) {
          const byPath = new Map<string, TreLookupRecord>()
          const uniqueTreFiles = new Map<string, File>()
          for (const entry of treIndexedEntries) {
            const key = `${entry.treName}|${entry.treFile.size}|${entry.treFile.lastModified}`
            if (!uniqueTreFiles.has(key)) uniqueTreFiles.set(key, entry.treFile)
          }

          const parsed = await Promise.allSettled(
            Array.from(uniqueTreFiles.values()).map(async (file) => ({
              file,
              parsed: await parseTreArchive(file),
            })),
          )

          for (const result of parsed) {
            if (result.status !== 'fulfilled') continue
            const treName = result.value.file.name
            for (const record of result.value.parsed.records) {
              const recordPath = normalizeSwgPath(record.name)
              const existing = byPath.get(recordPath)
              if (!existing) {
                byPath.set(recordPath, {
                  treName,
                  treFile: result.value.file,
                  record,
                })
                continue
              }

              if (treArchivePriority(treName) >= treArchivePriority(existing.treName)) {
                byPath.set(recordPath, {
                  treName,
                  treFile: result.value.file,
                  record,
                })
              }
            }
          }

          treLookupByPathRef.current = byPath
          treLookupBuiltRef.current = true
        }

        recordHit = treLookupByPathRef.current.get(normalized) ?? null
      }
    }

    if (!recordHit) {
      if (isTexture) console.error(`[File Lookup] \u2717 NOT FOUND in any source: ${normalized}`)
      return null
    }

    if (isTexture) console.log(`[File Lookup] \u2713 Extracting from TRE: ${normalized} (${recordHit.treName})`)
    const file = await extractTreRecord(recordHit.treFile, recordHit.record)
    const source = {
      file,
      sourceLabel: recordHit.treName,
    }
    treExtractedByPathRef.current.set(normalized, source)
    return source
  }, [indexedByPath, resolvedTreByPath, treIndexedEntries])

  useEffect(() => {
    if (!activeBuilding || activeBuilding.objects.length === 0) return

    let canceled = false
    const templatePaths = Array.from(
      new Set(
        [
          ...activeBuilding.objects
            .map((object) => object.templatePath)
            .filter((path): path is string => Boolean(path))
            .map((path) => normalizeSwgPath(path)),
          ...(activeBuilding.rootTemplatePath
            ? [normalizeSwgPath(activeBuilding.rootTemplatePath)]
            : []),
        ],
      ),
    )

    if (templatePaths.length === 0) return

    void (async () => {
      const updates: ResolvedVisualMap = {}
      for (const templatePath of templatePaths) {
        const existing = resolvedVisuals[templatePath]
        const isRootTemplate =
          Boolean(activeBuilding?.rootTemplatePath) &&
          templatePath === normalizeSwgPath(activeBuilding.rootTemplatePath as string)
        if (existing?.status === 'mesh' && !isRootTemplate) continue
        const visual = await resolveTemplateVisual(templatePath, lookupRepositoryFile, {
          strictDeclaredOnly: true,
          preferDecodedOverComposite: true,
        })
        updates[templatePath] = visual
      }

      if (canceled || Object.keys(updates).length === 0) return
      setResolvedVisuals((prev) => ({ ...prev, ...updates }))
      appendLog(`Resolved ${Object.keys(updates).length} template visual(s) for preview.`)

      if (activeBuilding?.rootTemplatePath) {
        const rootKey = normalizeSwgPath(activeBuilding.rootTemplatePath)
        const rootVisual = updates[rootKey]
        if (rootVisual) {
          const meshVerts = rootVisual.mesh
            ? Math.floor(rootVisual.mesh.positions.length / 3)
            : 0
          const meshTris = rootVisual.mesh
            ? Math.floor(rootVisual.mesh.indices.length / 3)
            : 0
          appendLog(
            `Root template ${rootVisual.status}: ${rootVisual.meshPath ?? 'no mesh'} [v=${meshVerts}, t=${meshTris}]`,
          )
        }
      }
    })()

    return () => {
      canceled = true
    }
  }, [activeBuilding, lookupRepositoryFile])

  useEffect(() => {
    const visuals = Object.entries(resolvedVisuals).filter((entry) => {
      const visual = entry[1]
      return Boolean(visual && (visual.texturePath || (visual.meshPartTexturePaths && visual.meshPartTexturePaths.length > 0)))
    })

    if (visuals.length === 0) return

    let canceled = false
    void (async () => {
      const next: ResolvedTextureMap = {}

      const textureCandidatesByFileName = new Map<string, string[]>()
      const pushTextureCandidate = (path: string) => {
        const normalized = normalizeSwgPath(path)
        const fileName = normalized.split('/').pop()
        if (!fileName) return
        const bucket = textureCandidatesByFileName.get(fileName) ?? []
        if (!bucket.includes(normalized)) {
          bucket.push(normalized)
          // Prefer canonical texture paths first when resolving by basename fallback.
          bucket.sort((a, b) => {
            const aTex = a.startsWith('texture/') ? 0 : 1
            const bTex = b.startsWith('texture/') ? 0 : 1
            if (aTex !== bTex) return aTex - bTex
            return a.localeCompare(b)
          })
          textureCandidatesByFileName.set(fileName, bucket)
        }
      }

      for (const path of indexedByPath.keys()) {
        if (path.endsWith('.dds') || path.endsWith('.tga')) pushTextureCandidate(path)
      }
      for (const path of resolvedTreByPath.keys()) {
        if (path.endsWith('.dds') || path.endsWith('.tga')) pushTextureCandidate(path)
      }

      const resolveTextureSource = async (
        rawPath: string,
      ): Promise<{ source: RepositorySourceFile; resolvedPath: string } | null> => {
        const normalizedRaw = normalizeSwgPath(rawPath)
        const directCandidates = [normalizedRaw]
        if (normalizedRaw.endsWith('.dds')) {
          directCandidates.push(`${normalizedRaw.slice(0, -4)}.tga`)
        } else if (normalizedRaw.endsWith('.tga')) {
          directCandidates.push(`${normalizedRaw.slice(0, -4)}.dds`)
        }

        for (const candidate of directCandidates) {
          const direct = await lookupRepositoryFile(candidate)
          if (direct) {
            return { source: direct, resolvedPath: candidate }
          }
        }

        const fileName = normalizedRaw.split('/').pop()
        if (fileName) {
          const byName = textureCandidatesByFileName.get(fileName) ?? []
          if (byName.length === 0) {
            // Fallback: include any matching texture records discovered via lazy TRE lookup map.
            for (const recordPath of treLookupByPathRef.current.keys()) {
              if (!recordPath.endsWith('.dds') && !recordPath.endsWith('.tga')) continue
              const recordFile = recordPath.split('/').pop()
              if (recordFile?.toLowerCase() !== fileName.toLowerCase()) continue
              pushTextureCandidate(recordPath)
            }
          }
          const expandedByName = textureCandidatesByFileName.get(fileName) ?? byName
          for (const candidate of expandedByName) {
            const hit = await lookupRepositoryFile(candidate)
            if (hit) return { source: hit, resolvedPath: candidate }
          }

          // One more pass: .dds/.tga sibling by filename in TRE lookup map when direct name misses.
          const altFileName = fileName.endsWith('.dds')
            ? `${fileName.slice(0, -4)}.tga`
            : fileName.endsWith('.tga')
              ? `${fileName.slice(0, -4)}.dds`
              : undefined
          if (altFileName) {
            for (const recordPath of treLookupByPathRef.current.keys()) {
              if (!recordPath.endsWith('.dds') && !recordPath.endsWith('.tga')) continue
              const recordFile = recordPath.split('/').pop()
              if (recordFile?.toLowerCase() !== altFileName.toLowerCase()) continue
              const hit = await lookupRepositoryFile(recordPath)
              if (hit) return { source: hit, resolvedPath: recordPath }
            }
          }
        }

        return null
      }

      const texturePathCandidates = (rawPath: string): string[] => {
        const lower = rawPath.toLowerCase()
        if (lower.endsWith('.dds')) return [rawPath, `${rawPath.slice(0, -4)}.tga`]
        if (lower.endsWith('.tga')) return [rawPath, `${rawPath.slice(0, -4)}.dds`]
        return [rawPath]
      }

      const findResolvedTextureByPath = (rawPath: string) => {
        const candidates = texturePathCandidates(rawPath).map((value) => value.toLowerCase())
        const pool = [
          ...Object.values(next),
          ...Object.values(resolvedTextures),
        ].filter((entry): entry is NonNullable<typeof entry> => Boolean(entry))

        for (const entry of pool) {
          const entryPath = entry.texturePath?.toLowerCase()
          if (!entryPath) continue
          if (candidates.includes(entryPath)) return entry
        }
        return undefined
      }

      const enqueueTexture = async (
        keyPath: string,
        texturePath: string,
        normalTexturePath?: string,
        secondaryTexturePath?: string,
        textureAddressU?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce',
        textureAddressV?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce',
        textureMipmapFilter?: 'none' | 'point' | 'linear' | 'anisotropic' | 'flatcubic' | 'gaussiancubic',
        textureMinificationFilter?: 'none' | 'point' | 'linear' | 'anisotropic' | 'flatcubic' | 'gaussiancubic',
        textureMagnificationFilter?: 'none' | 'point' | 'linear' | 'anisotropic' | 'flatcubic' | 'gaussiancubic',
        normalTextureAddressU?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce',
        normalTextureAddressV?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce',
        secondaryTextureAddressU?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce',
        secondaryTextureAddressV?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce',
        primaryUvSetIndex?: number,
        secondaryUvSetIndex?: number,
      ) => {
        // Normalise empty string → undefined so the equality check is stable.
        // The resolver pushes '' for parts with no normal map; enqueueTexture stores undefined.
        // Without this, '' !== undefined causes an infinite re-resolve loop.
        const normalPathNorm = normalTexturePath || undefined
        const secondaryPathNorm = secondaryTexturePath || undefined
        const existingTexture = resolvedTextures[keyPath]
        if (
          existingTexture?.texturePath === texturePath &&
          existingTexture?.normalTexturePath === normalPathNorm &&
          existingTexture?.secondaryTexturePath === secondaryPathNorm &&
          existingTexture?.textureAddressU === textureAddressU &&
          existingTexture?.textureAddressV === textureAddressV &&
          existingTexture?.textureMipmapFilter === textureMipmapFilter &&
          existingTexture?.textureMinificationFilter === textureMinificationFilter &&
          existingTexture?.textureMagnificationFilter === textureMagnificationFilter &&
          existingTexture?.normalTextureAddressU === normalTextureAddressU &&
          existingTexture?.normalTextureAddressV === normalTextureAddressV &&
          existingTexture?.secondaryTextureAddressU === secondaryTextureAddressU &&
          existingTexture?.secondaryTextureAddressV === secondaryTextureAddressV
        ) return

        const textureSource = await resolveTextureSource(texturePath)
        if (!textureSource) {
          const reused = findResolvedTextureByPath(texturePath)
          if (!reused) return
          next[keyPath] = {
            ...reused,
            textureAddressU,
            textureAddressV,
            textureMipmapFilter,
            textureMinificationFilter,
            textureMagnificationFilter,
            primaryUvSetIndex,
            secondaryUvSetIndex,
          }
          return
        }
        const resolvedTexturePath = textureSource.resolvedPath
        const hit = textureSource.source

        if (existingTexture) {
          URL.revokeObjectURL(existingTexture.url)
          if (existingTexture.alternateUrl) {
            URL.revokeObjectURL(existingTexture.alternateUrl)
          }
          if (existingTexture.normalUrl) {
            URL.revokeObjectURL(existingTexture.normalUrl)
          }
          if (existingTexture.secondaryUrl) {
            URL.revokeObjectURL(existingTexture.secondaryUrl)
          }
        }

        const cacheKey = `${keyPath}|${texturePath}`
        let url = objectUrlsRef.current.get(cacheKey)
        if (!url) {
          url = URL.createObjectURL(hit.file)
          objectUrlsRef.current.set(cacheKey, url)
        }

        let alternateUrl: string | undefined
        let resolvedAlternateTexturePath: string | undefined
        const normalizedResolvedTexturePath = resolvedTexturePath.toLowerCase()
        for (const altCandidate of texturePathCandidates(texturePath)) {
          const altNormalized = altCandidate.toLowerCase()
          if (altNormalized === normalizedResolvedTexturePath) continue
          const altSource = await resolveTextureSource(altCandidate)
          if (!altSource) continue
          if (altSource.resolvedPath.toLowerCase() === normalizedResolvedTexturePath) continue
          resolvedAlternateTexturePath = altSource.resolvedPath
          const altCacheKey = `${keyPath}|alt|${resolvedAlternateTexturePath}`
          alternateUrl = objectUrlsRef.current.get(altCacheKey)
          if (!alternateUrl) {
            alternateUrl = URL.createObjectURL(altSource.source.file)
            objectUrlsRef.current.set(altCacheKey, alternateUrl)
          }
          break
        }

        let normalUrl: string | undefined
        let resolvedNormalTexturePath: string | undefined
        if (normalTexturePath) {
          const normalSource = await resolveTextureSource(normalTexturePath)
          if (normalSource) {
            resolvedNormalTexturePath = normalSource.resolvedPath
            const normalHit = normalSource.source
            const normalCacheKey = `${keyPath}|normal|${resolvedNormalTexturePath}`
            normalUrl = objectUrlsRef.current.get(normalCacheKey)
            if (!normalUrl) {
              normalUrl = URL.createObjectURL(normalHit.file)
              objectUrlsRef.current.set(normalCacheKey, normalUrl)
            }
          }
        }

        let secondaryUrl: string | undefined
        let resolvedSecondaryTexturePath: string | undefined
        if (secondaryPathNorm) {
          const secondarySource = await resolveTextureSource(secondaryPathNorm)
          if (secondarySource) {
            resolvedSecondaryTexturePath = secondarySource.resolvedPath
            const secondaryHit = secondarySource.source
            const secondaryCacheKey = `${keyPath}|secondary|${resolvedSecondaryTexturePath}`
            secondaryUrl = objectUrlsRef.current.get(secondaryCacheKey)
            if (!secondaryUrl) {
              secondaryUrl = URL.createObjectURL(secondaryHit.file)
              objectUrlsRef.current.set(secondaryCacheKey, secondaryUrl)
            }
          }
        }

        next[keyPath] = {
          url,
          texturePath: resolvedTexturePath,
          alternateUrl,
          alternateTexturePath: alternateUrl ? resolvedAlternateTexturePath : undefined,
          textureAddressU,
          textureAddressV,
          textureMipmapFilter,
          textureMinificationFilter,
          textureMagnificationFilter,
          normalUrl,
          normalTexturePath: normalUrl ? resolvedNormalTexturePath : undefined,
          normalTextureAddressU: normalUrl ? normalTextureAddressU : undefined,
          normalTextureAddressV: normalUrl ? normalTextureAddressV : undefined,
          secondaryUrl,
          secondaryTexturePath: secondaryUrl ? resolvedSecondaryTexturePath : undefined,
          secondaryTextureAddressU: secondaryUrl ? secondaryTextureAddressU : undefined,
          secondaryTextureAddressV: secondaryUrl ? secondaryTextureAddressV : undefined,
          primaryUvSetIndex,
          secondaryUvSetIndex,
        }
      }

      for (const [templatePath, visualMaybe] of visuals) {
        const visual = visualMaybe as ResolvedTemplateVisual

        if (visual.texturePath) {
          await enqueueTexture(
            templatePath,
            visual.texturePath,
            visual.normalTexturePath,
            undefined,
            visual.textureAddressU,
            visual.textureAddressV,
            visual.textureMipmapFilter,
            visual.textureMinificationFilter,
            visual.textureMagnificationFilter,
            visual.normalTextureAddressU,
            visual.normalTextureAddressV,
            undefined,
            undefined,
          )
        }

        const partTexturePaths = visual.meshPartTexturePaths ?? []
        const partNormalTexturePaths = visual.meshPartNormalTexturePaths ?? []
        for (let partIndex = 0; partIndex < partTexturePaths.length; partIndex += 1) {
          const partTexturePath = partTexturePaths[partIndex]
          if (!partTexturePath) continue
          const partTextureKey = buildPartTextureKey(templatePath, partIndex)
          await enqueueTexture(
            partTextureKey,
            partTexturePath,
            partNormalTexturePaths[partIndex],
            visual.meshPartSecondaryTexturePaths?.[partIndex],
            visual.meshPartTextureAddressU?.[partIndex],
            visual.meshPartTextureAddressV?.[partIndex],
            visual.meshPartTextureMipmapFilter?.[partIndex],
            visual.meshPartTextureMinificationFilter?.[partIndex],
            visual.meshPartTextureMagnificationFilter?.[partIndex],
            visual.meshPartNormalTextureAddressU?.[partIndex],
            visual.meshPartNormalTextureAddressV?.[partIndex],
            visual.meshPartSecondaryTextureAddressU?.[partIndex],
            visual.meshPartSecondaryTextureAddressV?.[partIndex],
            visual.meshPartPrimaryUvSetIndices?.[partIndex],
            visual.meshPartSecondaryUvSetIndices?.[partIndex],
          )
        }
      }

      if (canceled || Object.keys(next).length === 0) return
      setResolvedTextures((prev) => ({ ...prev, ...next }))
      appendLog(`Resolved ${Object.keys(next).length} texture asset(s) for preview materials.`)
    })()

    return () => {
      canceled = true
    }
  }, [indexedByPath, lookupRepositoryFile, resolvedTextures, resolvedTreByPath, resolvedVisuals])

  useEffect(
    () => () => {
      objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url))
      objectUrlsRef.current.clear()
    },
    [],
  )

  function appendLog(message: string): void {
    setLog((prev) => [message, ...prev].slice(0, 20))
  }

  async function applyRootMeshOverride(path: string): Promise<void> {
    const normalized = normalizeSwgPath(path)
    if (!normalized || !activeBuilding?.rootTemplatePath) return

    const rootTemplate = normalizeSwgPath(activeBuilding.rootTemplatePath)
    const rootVisual = resolvedVisuals[rootTemplate]
    if (!rootVisual) {
      appendLog('Root mesh override skipped: root visual has not resolved yet.')
      return
    }

    const source = await lookupRepositoryFile(normalized)
    if (!source) {
      appendLog(`Root mesh override missing: ${normalized}`)
      return
    }

    const buffer = await source.file.arrayBuffer()
    const mesh = extractMeshFromAsset(buffer, normalized)
    if (!mesh) {
      appendLog(`Root mesh override decode failed: ${normalized}`)
      return
    }

    setResolvedVisuals((prev) => ({
      ...prev,
      [rootTemplate]: {
        ...rootVisual,
        status: 'mesh',
        meshPath: normalized,
        sourceLabel: source.sourceLabel,
        mesh,
        meshParts: undefined,
      },
    }))

    const verts = Math.floor(mesh.positions.length / 3)
    const tris = Math.floor(mesh.indices.length / 3)
    appendLog(`Root mesh override applied: ${normalized} [v=${verts}, t=${tris}]`)
  }

  function clearRootMeshOverride(): void {
    setRootMeshOverridePath('')
    appendLog('Root mesh override cleared. Re-select the building to re-resolve default chain.')
  }

  function activateAsset(asset: AssetRecord): void {
    if (asset.kind === 'building') {
      const linkedBuilding = buildings.find((b) => b.sourceAssetId === asset.id)
      if (linkedBuilding) {
        setActiveBuildingId(linkedBuilding.id)
        setSelectedObjectId(null)
      }
    } else if (asset.kind === 'object') {
      setSelectedObjectAssetId(asset.id)
    }
    appendLog(`Loaded asset: ${asset.relativePath}`)
  }

  function ingestEntries(
    entries: Array<{ file: File; relativePath: string; kind: AssetKind; source: 'loose' | 'tre' }>,
  ): void {
    if (entries.length === 0) return

    const dedupe = new Set(assets.map((a) => `${a.kind}|${a.relativePath}`))
    const nextAssets: AssetRecord[] = []
    const nextBuildings: BuildingDocument[] = []

    for (const entry of entries) {
      const key = `${entry.kind}|${entry.relativePath}`
      if (dedupe.has(key)) continue
      dedupe.add(key)

      const id = newId(entry.kind)
      const asset: AssetRecord = {
        id,
        name: entry.file.name,
        kind: entry.kind,
        size: entry.file.size,
        lastModified: entry.file.lastModified,
        relativePath: entry.relativePath,
        source: entry.source,
      }
      nextAssets.push(asset)
      looseFileByPathRef.current.set(normalizeSwgPath(entry.relativePath), entry.file)

      if (entry.kind === 'building') {
        entry.file.arrayBuffer().then((buffer) => {
          const iff = parseIff(buffer)
          const extracted = extractBuildingObjectsFromIff(buffer)
          const inferredWidth = extracted.length
            ? clamp(
                Math.max(...extracted.map((o) => Math.abs(o.x))) * 2 + 6,
                6,
                200,
              )
            : 36
          const inferredDepth = extracted.length
            ? clamp(
                Math.max(...extracted.map((o) => Math.abs(o.z))) * 2 + 6,
                6,
                200,
              )
            : 28
          const inferredHeight = extracted.length
            ? clamp(
                Math.max(...extracted.map((o) => Math.abs(o.y))) + 8,
                6,
                200,
              )
            : 16

          setBuildings((prev) =>
            prev.map((item) => {
              if (item.sourceAssetId !== id) return item
              return {
                ...item,
                width: inferredWidth,
                depth: inferredDepth,
                height: inferredHeight,
                objects: extracted.map((obj, index) => ({
                  id: newId('placed'),
                  name: obj.templatePath.split('/').pop() ?? `obj_${index + 1}`,
                  sourceAssetId: id,
                  templatePath: obj.templatePath,
                  x: obj.x,
                  y: obj.y,
                  z: obj.z,
                  yaw: obj.yaw,
                  scale: 1,
                })),
              }
            }),
          )

          appendLog(`Parsed IFF container: ${entry.file.name} [${iff.tag}] ${iff.size} bytes`)
          if (extracted.length === 0) {
            appendLog(`No object placements discovered in ${entry.file.name}.`)
          } else {
            appendLog(`Extracted ${extracted.length} object placement(s) from ${entry.file.name}.`)
          }
        })
        nextBuildings.push({
          id: newId('buildingDoc'),
          name: entry.file.name,
          rootTemplatePath: normalizeSwgPath(entry.relativePath),
          sourceAssetId: id,
          width: 36,
          depth: 28,
          height: 16,
          objects: [],
        })
      }
    }

    if (nextAssets.length === 0) return

    setAssets((prev) => [...nextAssets, ...prev])

    if (nextBuildings.length > 0) {
      setBuildings((prev) => [...nextBuildings, ...prev])
      setActiveBuildingId((prev) => prev ?? nextBuildings[0].id)
      setSelectedObjectId(null)
    }

    const firstObject = nextAssets.find((a) => a.kind === 'object')
    if (!selectedObjectAssetId && firstObject) {
      setSelectedObjectAssetId(firstObject.id)
    }

    appendLog(`Loaded ${nextAssets.length} asset file(s).`)
  }

  const ingestFiles = async (
    files: FileList | null,
    type: 'building' | 'object' | 'appearance',
  ) => {
    if (!files) return;

    const filePromises = Array.from(files).map(async (file) => {
      const content = await file.text();
      // Process file content asynchronously
      if (type === 'building') {
        processBuildingIFF(content);
      } else if (type === 'object') {
        processObjectIFF(content);
      } else if (type === 'appearance') {
        processAppearanceAssets(content);
      }
    });

    await Promise.all(filePromises);
    console.log(`${type} files loaded successfully.`);
  };

  const processBuildingIFF = (content: string) => {
    // Optimize parsing logic here
    console.log('Processing building IFF:', content.slice(0, 100))
  };

  const processObjectIFF = (content: string) => {
    // Optimize parsing logic here
    console.log('Processing object IFF:', content.slice(0, 100))
  };

  const processAppearanceAssets = (content: string) => {
    // Add logic to handle appearance assets
    console.log('Processing appearance assets:', content.slice(0, 100));
  };

  function loadIndexedFile(relativePath: string): void {
    const hit = indexedFiles.find((item) => item.relativePath === relativePath)
    if (!hit) {
      appendLog(`Indexed file not found: ${relativePath}`)
      return
    }

    const kind = inferAssetKind(hit.file.name, hit.relativePath)
    if (!kind) {
      appendLog(`Unsupported file type: ${hit.relativePath}`)
      return
    }

    const existing = assets.find((a) => a.kind === kind && a.relativePath === relativePath)
    if (existing) {
      activateAsset(existing)
      return
    }

    ingestEntries([
      {
        file: hit.file,
        relativePath,
        kind,
        source: 'loose',
      },
    ])
  }

  async function loadTreEntry(treEntryId: string): Promise<void> {
    const entry = treIndexedEntries.find((item) => item.id === treEntryId)
    if (!entry) return

    const relativePath = `repository/${entry.path}`
    const existing = assets.find((a) => a.relativePath === relativePath && a.kind === entry.kind)
    if (existing) {
      activateAsset(existing)
      return
    }

    setIndexingStatus(`Extracting ${entry.name} from ${entry.treName}...`)
    const extractedFile = await extractTreEntryFile(entry)
    setIndexingStatus(null)
    if (!extractedFile) {
      appendLog(`TRE entry could not be extracted: ${entry.path}`)
      return
    }

    ingestEntries([
      {
        file: extractedFile,
        relativePath,
        kind: entry.kind,
        source: 'tre',
      },
    ])
  }

  async function collectFilesFromDirectory(
    handle: FileSystemDirectoryHandleLike,
    relative = '',
  ): Promise<IndexedFileEntry[]> {
    const collected: IndexedFileEntry[] = []

    for await (const [entryName, entryHandle] of handle.entries()) {
      const nextPath = relative ? `${relative}/${entryName}` : entryName
      if (entryHandle.kind === 'directory') {
        const nested = await collectFilesFromDirectory(entryHandle, nextPath)
        collected.push(...nested)
      } else {
        const file = await entryHandle.getFile()
        collected.push({ file, relativePath: nextPath })
      }
    }

    return collected
  }

  async function indexDataRoot(): Promise<void> {
    const pickerWindow = window as WindowWithDirectoryPicker
    if (!pickerWindow.showDirectoryPicker) {
      appendLog('Directory picker is not available in this browser context.')
      return
    }

    try {
      setIndexingStatus('Waiting for folder selection...')
      const root = await pickerWindow.showDirectoryPicker()
      setIndexingStatus(`Indexing ${root.name}...`)

      const files = await collectFilesFromDirectory(root)
      setIndexedFiles(files)

      const recognized = files
        .map((item) => ({ item, kind: inferAssetKind(item.file.name, item.relativePath) }))
        .filter((row): row is { item: IndexedFileEntry; kind: AssetKind } => row.kind !== null)

      ingestEntries(
        recognized.map((row) => ({
          file: row.item.file,
          relativePath: row.item.relativePath,
          kind: row.kind,
          source: 'loose',
        })),
      )

      appendLog(`Indexed ${recognized.length} supported assets from ${root.name}.`)
      setIndexingStatus(null)
    } catch {
      setIndexingStatus(null)
      appendLog('Indexing canceled or failed.')
    }
  }

  async function ingestTreFiles(files: FileList | null): Promise<void> {
    if (!files || files.length === 0) return

    setIndexingStatus('Indexing TRE files...')
    const incoming = Array.from(files)
    const indexed: TreIndexedEntry[] = []

    const results = await Promise.allSettled(
      incoming.map(async (file) => {
        const entries = await indexTreFile(file)
        return { fileName: file.name, entries }
      }),
    )

    for (const result of results) {
      if (result.status === 'fulfilled') {
        indexed.push(...result.value.entries)
        appendLog(
          `Indexed TRE ${result.value.fileName}: ${result.value.entries.length} records parsed.`,
        )
      } else {
        appendLog('Failed to index TRE file.')
      }
    }

    setTreIndexedEntries((prev) => {
      const seen = new Set(prev.map((e) => `${e.treName}|${e.path}`))
      const next = [...prev]
      for (const entry of indexed) {
        const key = `${entry.treName}|${entry.path}`
        if (seen.has(key)) continue
        seen.add(key)
        next.push(entry)
      }
      return next
    })

    treLookupBuiltRef.current = false
    treLookupByPathRef.current.clear()
    treExtractedByPathRef.current.clear()

    setIndexingStatus(null)
  }

  function updateBuildingMetric(
    key: 'width' | 'depth' | 'height',
    value: number,
  ): void {
    if (!activeBuilding) return

    const normalized = clamp(value, 6, 200)
    setBuildings((prev) =>
      prev.map((item) =>
        item.id === activeBuilding.id ? { ...item, [key]: normalized } : item,
      ),
    )
  }

  function addObjectToActiveBuilding(): void {
    if (!activeBuilding) return
    const source =
      objectAssets.find((asset) => asset.id === selectedObjectAssetId) ?? objectAssets[0]
    if (!source) {
      appendLog('No object IFF files loaded yet.')
      return
    }

    const appearance = appearanceAssets[0]
    const object: PlacedObject = {
      id: newId('placed'),
      name: source.name,
      sourceAssetId: source.id,
      templatePath: normalizeSwgPath(source.relativePath),
      x: 0,
      y: 0,
      z: 0,
      yaw: 0,
      scale: 1,
      appearanceAssetId: appearance?.id,
    }

    setBuildings((prev) =>
      prev.map((item) =>
        item.id === activeBuilding.id
          ? { ...item, objects: [...item.objects, object] }
          : item,
      ),
    )
    setSelectedObjectId(object.id)
    appendLog(`Added ${source.name} to ${activeBuilding.name}.`)
  }

  function removeSelectedObject(): void {
    if (!activeBuilding || !selectedObjectId) return

    setBuildings((prev) =>
      prev.map((item) =>
        item.id === activeBuilding.id
          ? {
              ...item,
              objects: item.objects.filter((object) => object.id !== selectedObjectId),
            }
          : item,
      ),
    )
    appendLog('Removed selected object from building definition.')
    setSelectedObjectId(null)
  }

  function updateObject(
    objectId: string,
    key: 'x' | 'z' | 'y' | 'yaw' | 'scale',
    value: number,
  ): void {
    if (!activeBuilding) return

    setBuildings((prev) =>
      prev.map((building) =>
        building.id !== activeBuilding.id
          ? building
          : {
              ...building,
              objects: building.objects.map((object) => {
                if (object.id !== objectId) return object
                const normalized =
                  key === 'scale'
                    ? clamp(value, 0.1, 12)
                    : key === 'yaw'
                      ? clamp(value, -360, 360)
                      : clamp(value, -200, 200)
                return { ...object, [key]: normalized }
              }),
            },
      ),
    )
  }

  function exportSession(): void {
    const payload = {
      exportedAt: new Date().toISOString(),
      assets,
      buildings,
    }

    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'swg-building-editor-session.json'
    anchor.click()
    URL.revokeObjectURL(url)

    appendLog('Exported session JSON snapshot.')
  }

  const selectedObject = activeBuilding?.objects.find(
    (object) => object.id === selectedObjectId,
  )
  const selectedObjectVisual = selectedObject?.templatePath
    ? resolvedVisuals[normalizeSwgPath(selectedObject.templatePath)]
    : undefined

  const formatAddressMode = (
    mode?: 'wrap' | 'mirror' | 'clamp' | 'border' | 'mirroronce',
  ): string => mode ?? 'n/a'

  const isInteriorBinding = (
    basePath: string,
    normalPath: string,
    detailPath: string,
    chunkTrace: string,
  ): boolean => {
    const hay = `${basePath} ${normalPath} ${detailPath} ${chunkTrace}`.toLowerCase()
    const isExteriorFamily =
      hay.includes('texture/thed_') ||
      hay.includes('texture/corellia_') ||
      hay.includes('texture/tato_') ||
      hay.includes('shader/thed_') ||
      hay.includes('shader/corellia_') ||
      hay.includes('shader/tato_') ||
      hay.includes('exterior') ||
      hay.includes('window') ||
      hay.includes('brick')
    if (isExteriorFamily) return false

    return (
      hay.includes('texture/nboo_intr_') ||
      hay.includes('texture/intr_') ||
      hay.includes('shader/nboo_intr_') ||
      hay.includes('shader/intr_')
    )
  }

  const buildPartBindingRows = (
    visual?: ResolvedTemplateVisual,
    options?: { interiorOnly?: boolean },
  ): string[] => {
    if (!visual) return []
    const maxParts = Math.max(
      visual.meshParts?.length ?? 0,
      visual.meshPartTexturePaths?.length ?? 0,
      visual.meshPartNormalTexturePaths?.length ?? 0,
      visual.meshPartSecondaryTexturePaths?.length ?? 0,
      visual.meshPartPrimaryUvSetIndices?.length ?? 0,
      visual.meshPartSecondaryUvSetIndices?.length ?? 0,
    )
    if (maxParts <= 0) return []

    const rows: string[] = []
    for (let i = 0; i < maxParts; i += 1) {
      const primaryUv = visual.meshPartPrimaryUvSetIndices?.[i]
      const secondaryUv = visual.meshPartSecondaryUvSetIndices?.[i]
      const basePath = visual.meshPartTexturePaths?.[i] || 'n/a'
      const normalPath = visual.meshPartNormalTexturePaths?.[i] || 'n/a'
      const detailPath = visual.meshPartSecondaryTexturePaths?.[i] || 'n/a'
      const baseAddrU = formatAddressMode(visual.meshPartTextureAddressU?.[i])
      const baseAddrV = formatAddressMode(visual.meshPartTextureAddressV?.[i])
      const normalAddrU = formatAddressMode(visual.meshPartNormalTextureAddressU?.[i])
      const normalAddrV = formatAddressMode(visual.meshPartNormalTextureAddressV?.[i])
      const detailAddrU = formatAddressMode(visual.meshPartSecondaryTextureAddressU?.[i])
      const detailAddrV = formatAddressMode(visual.meshPartSecondaryTextureAddressV?.[i])
      const chunkTrace = visual.meshPartChunkTrace?.[i] ?? 'n/a'

      if (options?.interiorOnly && !isInteriorBinding(basePath, normalPath, detailPath, chunkTrace)) {
        continue
      }

      rows.push(
        `part ${i}: uv=${primaryUv ?? 'n/a'} base=[${baseAddrU},${baseAddrV}] ${basePath} | normal=[${normalAddrU},${normalAddrV}] ${normalPath} | detailUv=${secondaryUv ?? 'n/a'} detail=[${detailAddrU},${detailAddrV}] ${detailPath} | trace=${chunkTrace}`,
      )
    }

    return rows
  }

  const selectedPartBindingRows = buildPartBindingRows(selectedObjectVisual)

  return (
    <div className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">SWG TOOLCHAIN</p>
          <h1>Building IFF Editor</h1>
        </div>
        <div className="topbar-actions">
          <label className="btn upload" htmlFor="building-upload">
            Load Building IFF
          </label>
          <input
            id="building-upload"
            name="building-upload"
            type="file"
            hidden
            accept=".iff"
            multiple
            onChange={(event) => ingestFiles(event.target.files, 'building')}
          />

          <label className="btn upload" htmlFor="object-upload">
            Load Object IFF
          </label>
          <input
            id="object-upload"
            name="object-upload"
            type="file"
            hidden
            accept=".iff"
            multiple
            onChange={(event) => ingestFiles(event.target.files, 'object')}
          />

          <label className="btn upload" htmlFor="appearance-upload">
            Load Appearance Assets
          </label>
          <input
            id="appearance-upload"
            name="appearance-upload"
            type="file"
            hidden
            accept=".iff,.apt,.msh,.lod"
            multiple
            onChange={(event) => ingestFiles(event.target.files, 'appearance')}
          />

          <label className="btn upload" htmlFor="tre-upload">
            Load TRE Files
          </label>
          <input
            id="tre-upload"
            name="tre-upload"
            type="file"
            hidden
            accept=".tre"
            multiple
            onChange={(event) => {
              void ingestTreFiles(event.target.files)
            }}
          />

          <button className="btn" type="button" onClick={indexDataRoot}>
            Index Data Root
          </button>
          <button className="btn" type="button" onClick={exportSession}>
            Export Session
          </button>
        </div>
      </header>

      <main className="workspace">
        <aside className="panel library">
          <h2>Asset Library</h2>
          <p className="muted">Index folders or TRE archives, then click a file to load/select it.</p>
          {indexingStatus && <p className="status">{indexingStatus}</p>}
          <input
            id="asset-search"
            name="asset-search"
            className="search"
            type="search"
            placeholder="Search assets by name/path"
            value={assetSearch}
            onChange={(event) => setAssetSearch(event.target.value)}
          />

          <div className="file-tree">
            {treeData.root.children && treeData.root.children.length > 0 ? (
              <FileTreeBrowser
                node={treeData.root}
                expandedFolders={expandedFolders}
                setExpandedFolders={setExpandedFolders}
                getTreSource={(path) => treSourceByTreePath.get(path)}
                onFileClick={(node) => {
                  const item = treeData.map.get(node.path)
                  if (!item) return

                  if (item.source === 'loaded') {
                    const asset = assets.find((a) => a.id === item.assetId)
                    if (asset) activateAsset(asset)
                    return
                  }

                  if (item.source === 'indexed') {
                    loadIndexedFile(item.relativePath)
                    return
                  }

                  void loadTreEntry(item.treEntryId)
                }}
              />
            ) : (
              <p className="empty">No files indexed yet.</p>
            )}
          </div>
        </aside>

        <section className="panel stage">
          <div className="stage-header">
            <div>
              <h2>Preview</h2>
              <p className="muted">
                {activeBuilding ? `Editing ${activeBuilding.name}` : 'No building loaded yet'}
              </p>
              {buildings.length > 0 && (
                <select
                  id="active-building-select"
                  name="active-building-select"
                  className="building-select"
                  value={activeBuildingId ?? ''}
                  onChange={(event) => {
                    setActiveBuildingId(event.target.value)
                    setSelectedObjectId(null)
                  }}
                >
                  {buildings.map((building) => (
                    <option key={building.id} value={building.id}>
                      {building.name}
                    </option>
                  ))}
                </select>
              )}
            </div>
            <div className="stage-actions">
              <select
                id="object-source-select"
                name="object-source-select"
                className="object-source"
                value={selectedObjectAssetId ?? ''}
                onChange={(event) => setSelectedObjectAssetId(event.target.value)}
              >
                {filteredObjectAssets.length === 0 && (
                  <option value="">No object assets loaded</option>
                )}
                {filteredObjectAssets.map((asset) => (
                  <option key={asset.id} value={asset.id}>
                    {asset.relativePath}
                  </option>
                ))}
              </select>
              <button className="btn" type="button" onClick={addObjectToActiveBuilding}>
                Add Object
              </button>
              <button className="btn danger" type="button" onClick={removeSelectedObject}>
                Remove Selected
              </button>
            </div>
          </div>

          <PreviewCanvas
            building={activeBuilding}
            selectedObjectId={selectedObjectId}
            resolvedVisuals={resolvedVisuals}
            resolvedTextures={resolvedTextures}
            renderTextures={renderTextures}
            interiorAssist={interiorAssist}
            interiorOpacity={interiorOpacity}
            roofCutaway={roofCutaway}
            roofCutawayLevel={roofCutawayLevel}
            showPlaceholderObjects={showPlaceholderObjects}
          />

          {activeBuilding && (
            <div className="build-metrics">
              <label>
                Width
                <input
                  id="building-width"
                  name="building-width"
                  type="number"
                  value={activeBuilding.width}
                  min={6}
                  max={200}
                  onChange={(event) => updateBuildingMetric('width', Number(event.target.value))}
                />
              </label>
              <label>
                Depth
                <input
                  id="building-depth"
                  name="building-depth"
                  type="number"
                  value={activeBuilding.depth}
                  min={6}
                  max={200}
                  onChange={(event) => updateBuildingMetric('depth', Number(event.target.value))}
                />
              </label>
              <label>
                Height
                <input
                  id="building-height"
                  name="building-height"
                  type="number"
                  value={activeBuilding.height}
                  min={6}
                  max={200}
                  onChange={(event) => updateBuildingMetric('height', Number(event.target.value))}
                />
              </label>
            </div>
          )}
        </section>

        <aside className="panel inspector">
          <h2>Inspector</h2>
          <p className="muted">Tune selected object transform and inspect loaded entries.</p>

          {activeBuilding?.objects.length ? (
            <ul className="object-list">
              {activeBuilding.objects.map((object) => (
                <li key={object.id}>
                  <button
                    type="button"
                    className={object.id === selectedObjectId ? 'object-pill active' : 'object-pill'}
                    onClick={() => setSelectedObjectId(object.id)}
                  >
                    {object.name}
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="empty">No objects in active building yet.</p>
          )}

          {selectedObject && (
            <div className="transform-grid">
              {([
                ['x', selectedObject.x],
                ['z', selectedObject.z],
                ['y', selectedObject.y],
                ['yaw', selectedObject.yaw],
                ['scale', selectedObject.scale],
              ] as const).map(([field, value]) => (
                <label key={field}>
                  {field.toUpperCase()}
                  <input
                    id={`object-${selectedObject.id}-${field}`}
                    name={`object-${selectedObject.id}-${field}`}
                    type="number"
                    value={value}
                    step={field === 'scale' ? 0.1 : 1}
                    onChange={(event) =>
                      updateObject(selectedObject.id, field, Number(event.target.value))
                    }
                  />
                </label>
              ))}
            </div>
          )}

          {selectedObjectVisual && (
            <section className="log-panel">
              <h3>Resolved Chain</h3>
              <ul>
                <li>Status: {selectedObjectVisual.status}</li>
                <li>Object: {selectedObjectVisual.objectPath ?? 'n/a'}</li>
                <li>Appearance: {selectedObjectVisual.appearancePath ?? 'n/a'}</li>
                <li>Shader: {selectedObjectVisual.shaderPath ?? 'n/a'}</li>
                <li>Effect: {selectedObjectVisual.effectPath ?? 'n/a'}</li>
                <li>Texture: {selectedObjectVisual.texturePath ?? 'n/a'}</li>
                <li>Normal: {selectedObjectVisual.normalTexturePath ?? 'n/a'}</li>
                <li>Effect options: {selectedObjectVisual.effectOptionCodes?.join(', ') ?? 'n/a'}</li>
                <li>Vertex programs: {selectedObjectVisual.vertexProgramPaths?.join(', ') ?? 'n/a'}</li>
                <li>Pixel programs: {selectedObjectVisual.pixelProgramPaths?.join(', ') ?? 'n/a'}</li>
                <li>Shader stage textures: {selectedObjectVisual.shaderStageTexturePaths?.join(', ') ?? 'n/a'}</li>
                <li>Shader stage normals: {selectedObjectVisual.shaderStageNormalTexturePaths?.join(', ') ?? 'n/a'}</li>
                <li>Mesh: {selectedObjectVisual.meshPath ?? 'n/a'}</li>
                <li>Source: {selectedObjectVisual.sourceLabel ?? 'n/a'}</li>
              </ul>
            </section>
          )}

          {selectedPartBindingRows.length > 0 && (
            <section className="log-panel">
              <h3>Selected Object Part Bindings</h3>
              <ul>
                {selectedPartBindingRows.map((row) => (
                  <li key={row}>{row}</li>
                ))}
              </ul>
            </section>
          )}

          {activeBuilding?.rootTemplatePath && (
            <section className="log-panel">
              <h3>Interior Assist</h3>
              <p className="muted">Makes the shell translucent and loosens orbit/pan so navigating inside is easier.</p>
              <label className="toggle-row">
                <span>Enable interior assist</span>
                <input
                  id="interior-assist-toggle"
                  name="interior-assist-toggle"
                  type="checkbox"
                  checked={interiorAssist}
                  onChange={(event) => setInteriorAssist(event.target.checked)}
                />
              </label>
              <label>
                Exterior opacity
                <input
                  id="exterior-opacity-range"
                  name="exterior-opacity-range"
                  type="range"
                  min={0.08}
                  max={1}
                  step={0.02}
                  value={interiorOpacity}
                  onChange={(event) => setInteriorOpacity(Number(event.target.value))}
                  disabled={!interiorAssist}
                />
              </label>
              <label className="toggle-row">
                <span>Enable roof cutaway</span>
                <input
                  id="roof-cutaway-toggle"
                  name="roof-cutaway-toggle"
                  type="checkbox"
                  checked={roofCutaway}
                  onChange={(event) => setRoofCutaway(event.target.checked)}
                />
              </label>
              <label>
                Cutaway height
                <input
                  id="roof-cutaway-level"
                  name="roof-cutaway-level"
                  type="range"
                  min={0.2}
                  max={1}
                  step={0.01}
                  value={roofCutawayLevel}
                  onChange={(event) => setRoofCutawayLevel(Number(event.target.value))}
                  disabled={!roofCutaway}
                />
              </label>
              <label className="toggle-row">
                <span>Render textures</span>
                <input
                  id="render-textures-toggle"
                  name="render-textures-toggle"
                  type="checkbox"
                  checked={renderTextures}
                  onChange={(event) => setRenderTextures(event.target.checked)}
                />
              </label>
              <label className="toggle-row">
                <span>Show unresolved placeholders</span>
                <input
                  id="show-placeholders-toggle"
                  name="show-placeholders-toggle"
                  type="checkbox"
                  checked={showPlaceholderObjects}
                  onChange={(event) => setShowPlaceholderObjects(event.target.checked)}
                />
              </label>
            </section>
          )}

          {activeBuilding?.rootTemplatePath && (
            <section className="log-panel">
              <h3>Root Mesh Override</h3>
              <p className="muted">Force a root mesh path for live debugging when resolver picks wrong parts.</p>
              <label>
                Mesh path
                <input
                  id="root-mesh-override-path"
                  name="root-mesh-override-path"
                  type="text"
                  value={rootMeshOverridePath}
                  placeholder="appearance/mesh/... .msh"
                  onChange={(event) => setRootMeshOverridePath(event.target.value)}
                />
              </label>
              {rootMeshOverrideOptions.length > 0 && (
                <label>
                  Candidate paths
                  <select
                    id="root-mesh-override-candidates"
                    name="root-mesh-override-candidates"
                    value=""
                    onChange={(event) => {
                      const next = event.target.value
                      if (!next) return
                      setRootMeshOverridePath(next)
                    }}
                  >
                    <option value="">Select candidate mesh path...</option>
                    {rootMeshOverrideOptions.map((path) => (
                      <option key={path} value={path}>{path}</option>
                    ))}
                  </select>
                </label>
              )}
              <div className="actions-row">
                <button
                  className="btn"
                  type="button"
                  onClick={() => {
                    void applyRootMeshOverride(rootMeshOverridePath)
                  }}
                  disabled={!rootMeshOverridePath.trim()}
                >
                  Apply Override
                </button>
                <button className="btn" type="button" onClick={clearRootMeshOverride}>
                  Clear
                </button>
              </div>
            </section>
          )}

          <section className="log-panel">
            <h3>Activity</h3>
            <ul>
              {log.map((entry, index) => (
                <li key={`${entry}_${index}`}>{entry}</li>
              ))}
            </ul>
          </section>

          <section className="log-panel">
            <h3>Sources</h3>
            <ul>
              <li>Loaded assets: {assets.length}</li>
              <li>Indexed files: {indexedFiles.length}</li>
              <li>TRE indexed entries: {treIndexedEntries.length}</li>
              <li>TRE resolved repository entries: {resolvedTreEntries.length}</li>
              <li>Resolved preview templates: {Object.keys(resolvedVisuals).length}</li>
              <li>Resolved preview textures: {Object.keys(resolvedTextures).length}</li>
            </ul>
          </section>

          <section className="log-panel">
            <h3>Tips</h3>
            <ul>
              <li>Clicking indexed folder files now loads them into the editor.</li>
              <li>Repository files show the winning TRE archive as a tag in the asset tree.</li>
              <li>Sizes shown in loaded assets use binary KB ({formatBytes(1024)}).</li>
            </ul>
          </section>
        </aside>
      </main>
    </div>
  )
}

export default App
