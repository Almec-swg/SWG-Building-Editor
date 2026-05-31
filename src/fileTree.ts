// File tree structure for asset browser
export interface FileTreeNode {
  name: string
  path: string
  kind: 'file' | 'folder'
  children?: FileTreeNode[]
  assetKind?: 'building' | 'object' | 'appearance' | null
}

export function buildFileTree(files: Array<{ file: File; relativePath: string }>): FileTreeNode {
  const root: FileTreeNode = { name: '', path: '', kind: 'folder', children: [] }
  const nodeMap = new Map<string, FileTreeNode>();
  nodeMap.set('', root);

  for (const { relativePath } of files) {
    const parts = relativePath.split(/[\\\/]/)
    let currentPath = ''

    for (let i = 0; i < parts.length; ++i) {
      const part = parts[i]
      const isFile = i === parts.length - 1
      currentPath = currentPath ? `${currentPath}/${part}` : part

      if (!nodeMap.has(currentPath)) {
        const parentPath = currentPath.substring(0, currentPath.lastIndexOf('/'))
        const parentNode = nodeMap.get(parentPath)

        const newNode: FileTreeNode = {
          name: part,
          path: currentPath,
          kind: isFile ? 'file' : 'folder',
          children: isFile ? undefined : [],
          assetKind: isFile ? undefined : null,
        }

        parentNode?.children?.push(newNode)
        if (!isFile) {
          nodeMap.set(currentPath, newNode)
        }
      }
    }
  }

  return root
}
