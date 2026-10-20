//! FBX (ASCII) import/export.
//!
//! FBX is a documented, human-readable text format: a tree of nodes written
//! `Name: prop, prop { … }`, arrays as a `*count` followed by an `a:` list
//! (writers wrap long arrays over several lines), strings in quotes, comments
//! after a `;`. This is a hand-written reader for that tree — the same spirit
//! as the COLLADA reader — carrying the parts of the schema a mesh needs:
//! `Objects` → `Geometry` (`Vertices`, `PolygonVertexIndex`, and the
//! `LayerElementNormal`, `LayerElementUV` and `LayerElementMaterial` layers).
//!
//! Two details of the schema matter here. The last corner of every polygon is
//! written as its bitwise complement — a negative number (`~3` is `-4`) — and
//! that is what says where a polygon ends, n-gons included; and a layer names
//! its values either per polygon vertex or per control point, directly or
//! through an index array. Both are read. A file with no normals gets smooth
//! ones like every other import. `Model` transforms, the `Materials` section
//! and its connections, and animation are out of scope, as they are for
//! COLLADA.
//!
//! Export writes one Y-up geometry per mesh: `Vertices`,
//! `PolygonVertexIndex` (the mesh's own n-gons, each closed with a
//! complement), a `ByPolygonVertex` normal layer, a `ByPolygonVertex` UV
//! layer when the mesh has UVs, a material layer (`AllSame`, or `ByPolygon`
//! when the faces carry different slots) and one `Model` connected to the
//! geometry.

use crate::{finalize_mesh, IoError, IoResult, LoadedScene, MeshData};
use openforge_core::mesh::{Corner, Face, Mesh};
use openforge_core::{Mat4, Vec2, Vec3};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

fn parse_err(message: impl Into<String>) -> IoError {
    IoError::Parse { format: "fbx", message: message.into() }
}

// ---------------------------------------------------------------------------
// Properties and the node tree.
// ---------------------------------------------------------------------------

/// One property on a node's header line, or one value of an `a:` list.
#[derive(Debug, Clone, PartialEq)]
enum Prop {
    Num(f64),
    Text(String),
}

/// A node of the FBX tree: a name, the properties on its own line, and its
/// children (empty unless the line opened a block).
#[derive(Debug, Default)]
struct Node {
    name: String,
    props: Vec<Prop>,
    children: Vec<Node>,
}

impl Node {
    fn child(&self, name: &str) -> Option<&Node> {
        self.children.iter().find(|child| child.name == name)
    }

    /// The first text property, e.g. a geometry's name or a mapping type.
    fn text(&self) -> Option<&str> {
        self.props.iter().find_map(|prop| match prop {
            Prop::Text(text) => Some(text.as_str()),
            Prop::Num(_) => None,
        })
    }

    /// The numeric properties, in order.
    fn numbers(&self) -> Vec<f64> {
        self.props
            .iter()
            .filter_map(|prop| match prop {
                Prop::Num(number) => Some(*number),
                Prop::Text(_) => None,
            })
            .collect()
    }

    /// An array node's values: its `a:` child when it has one (the usual
    /// shape), or its own numbers.
    fn array(&self) -> Vec<f64> {
        match self.child("a") {
            Some(values) => values.numbers(),
            None => self.numbers(),
        }
    }
}

/// Drop a `;` comment, unless the `;` sits inside a quoted string.
fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    let mut escaped = false;
    for (at, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            ';' if !in_string => return &line[..at],
            _ => {}
        }
    }
    line
}

/// The file's lines with comments dropped and wrapped value lists joined back
/// onto the line they belong to: a line that carries no `:` of its own is a
/// continuation of the array above it.
fn normalized_lines(text: &str) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for raw in text.lines() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        let starts_something = line.contains(':') || line == "}" || line == "{";
        match lines.last_mut() {
            Some(last) if !starts_something => {
                last.push(' ');
                last.push_str(line);
            }
            _ => lines.push(line.to_string()),
        }
    }
    lines
}

/// Split a property list on commas, keeping quoted strings whole.
fn split_props(text: &str) -> Vec<Prop> {
    let mut props = Vec::new();
    let mut token = String::new();
    let mut quoted = false;
    let mut in_string = false;
    let mut escaped = false;
    for ch in text.chars() {
        if escaped {
            token.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string => escaped = true,
            '"' => {
                in_string = !in_string;
                quoted = true;
            }
            ',' if !in_string => {
                push_prop(&mut props, &token, quoted);
                token.clear();
                quoted = false;
            }
            _ => token.push(ch),
        }
    }
    push_prop(&mut props, &token, quoted);
    props
}

fn push_prop(props: &mut Vec<Prop>, token: &str, quoted: bool) {
    let token = token.trim();
    if quoted {
        props.push(Prop::Text(token.to_string()));
        return;
    }
    if token.is_empty() || token.starts_with('*') {
        // Empty, or the `*count` marker: a length, not a value.
        return;
    }
    // `~3` is the complement of 3, the same -4 some writers spell out.
    if let Some(rest) = token.strip_prefix('~') {
        if let Ok(number) = rest.trim().parse::<f64>() {
            props.push(Prop::Num(-number - 1.0));
            return;
        }
    }
    match token.parse::<f64>() {
        Ok(number) => props.push(Prop::Num(number)),
        Err(_) => props.push(Prop::Text(token.to_string())),
    }
}

/// Read the node tree. A block left open at the end of the file is the one
/// structural refusal: a truncated file, not a document to guess at.
fn parse_tree(text: &str) -> IoResult<Vec<Node>> {
    let mut roots: Vec<Node> = Vec::new();
    let mut stack: Vec<Node> = Vec::new();
    for line in normalized_lines(text) {
        if line == "}" {
            let node = stack.pop().ok_or_else(|| parse_err("a closing brace with no block open"))?;
            match stack.last_mut() {
                Some(parent) => parent.children.push(node),
                None => roots.push(node),
            }
            continue;
        }
        let Some((name, rest)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_string();
        if name.is_empty() {
            return Err(parse_err("a line starts with ':'"));
        }
        let rest = rest.trim();
        // A `{` opens a block. Anything after it — a writer that kept an
        // array on the header line — is this node's values, and a `}` on the
        // same line closes the block again.
        let mut opens = false;
        let mut closed_inline = false;
        let mut inline_values = String::new();
        let head = match rest.split_once('{') {
            Some((head, tail)) => {
                opens = true;
                let tail = tail.trim();
                closed_inline = tail.ends_with('}');
                let body = tail.trim_end_matches('}').trim();
                let values = body.strip_prefix("a:").unwrap_or(body).trim();
                inline_values = values.to_string();
                head.trim().to_string()
            }
            None => rest.to_string(),
        };
        let mut node = Node { name, props: split_props(&head), children: Vec::new() };
        if !inline_values.is_empty() {
            node.children.push(Node {
                name: "a".to_string(),
                props: split_props(&inline_values),
                children: Vec::new(),
            });
        }
        if opens && !closed_inline {
            stack.push(node);
        } else {
            match stack.last_mut() {
                Some(parent) => parent.children.push(node),
                None => roots.push(node),
            }
        }
    }
    if let Some(open) = stack.last() {
        return Err(parse_err(format!("unterminated `{}` block", open.name)));
    }
    Ok(roots)
}

// ---------------------------------------------------------------------------
// Layers.
// ---------------------------------------------------------------------------

/// How a layer's values line up with the polygons.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mapping {
    /// One value per polygon vertex (in `PolygonVertexIndex` order).
    PolygonVertex,
    /// One value per control point (position index).
    ControlPoint,
}

/// Whether a corner takes a value directly or through the layer's index array.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reference {
    Direct,
    IndexToDirect,
}

/// One `LayerElement*`: its raw components and how a corner picks one.
struct Layer {
    mapping: Mapping,
    reference: Reference,
    values: Vec<f64>,
    indices: Vec<f64>,
    components: usize,
}

impl Layer {
    /// The value index for one corner: `corner` is the running polygon-vertex
    /// counter, `point` the control point's index.
    fn value_index(&self, corner: usize, point: usize) -> IoResult<usize> {
        let index = match (self.mapping, self.reference) {
            (Mapping::PolygonVertex, Reference::Direct) => corner,
            (Mapping::PolygonVertex, Reference::IndexToDirect) => self.index_at(corner)?,
            (Mapping::ControlPoint, Reference::Direct) => point,
            (Mapping::ControlPoint, Reference::IndexToDirect) => self.index_at(point)?,
        };
        let count = self.values.len() / self.components;
        if index >= count {
            return Err(parse_err(format!("a layer names entry {index} of {count}")));
        }
        Ok(index)
    }

    fn index_at(&self, at: usize) -> IoResult<usize> {
        match self.indices.get(at) {
            Some(value) if *value >= 0.0 && value.fract() == 0.0 => Ok(*value as usize),
            Some(value) => {
                Err(parse_err(format!("a layer index is not a whole number: {value}")))
            }
            None => Err(parse_err(format!("a layer's index array has no entry {at}"))),
        }
    }

    fn value3(&self, index: usize) -> Vec3 {
        let base = index * 3;
        Vec3::new(self.values[base] as f32, self.values[base + 1] as f32, self.values[base + 2] as f32)
    }

    fn value2(&self, index: usize) -> Vec2 {
        let base = index * 2;
        Vec2::new(self.values[base] as f32, self.values[base + 1] as f32)
    }
}

/// Read one layer, when the geometry carries it. An empty value list reads as
/// no layer at all, the way a writer that has no UVs leaves the layer behind.
fn read_layer(
    geometry: &Node,
    layer: &str,
    value_key: &str,
    index_key: &str,
    components: usize,
) -> IoResult<Option<Layer>> {
    let Some(node) = geometry.child(layer) else {
        return Ok(None);
    };
    let Some(values) = node.child(value_key) else {
        return Err(parse_err(format!("{layer} has no {value_key}")));
    };
    let values = values.array();
    if values.is_empty() {
        return Ok(None);
    }
    if values.len() % components != 0 {
        return Err(parse_err(format!(
            "{layer}'s {value_key} has {} numbers, not a multiple of {components}",
            values.len()
        )));
    }
    let mapping = match node.child("MappingInformationType").and_then(Node::text) {
        Some("ByControlPoint") | Some("ByVertice") => Mapping::ControlPoint,
        _ => Mapping::PolygonVertex,
    };
    let reference = match node.child("ReferenceInformationType").and_then(Node::text) {
        Some("IndexToDirect") => Reference::IndexToDirect,
        _ => Reference::Direct,
    };
    let indices = node.child(index_key).map(Node::array).unwrap_or_default();
    Ok(Some(Layer { mapping, reference, values, indices, components }))
}

// ---------------------------------------------------------------------------
// Geometry.
// ---------------------------------------------------------------------------

/// Split the corner list into polygons. The last corner of each polygon is
/// written as its bitwise complement — a negative number — so that is where
/// one polygon ends and the next begins, n-gons included.
fn polygons(indices: &[f64], vertex_count: usize, name: &str) -> IoResult<Vec<Vec<u32>>> {
    let mut polygons: Vec<Vec<u32>> = Vec::new();
    let mut current: Vec<u32> = Vec::new();
    for &value in indices {
        if value.fract() != 0.0 {
            return Err(parse_err(format!(
                "`{name}`: PolygonVertexIndex holds {value}, not a whole number"
            )));
        }
        let last = value < 0.0;
        let index = if last { (-value - 1.0) as i64 } else { value as i64 };
        if index < 0 || index as usize >= vertex_count {
            return Err(parse_err(format!(
                "`{name}`: PolygonVertexIndex names vertex {index} of {vertex_count}"
            )));
        }
        current.push(index as u32);
        if last {
            if current.len() < 3 {
                return Err(parse_err(format!(
                    "`{name}`: a polygon has {} corners, need at least 3",
                    current.len()
                )));
            }
            polygons.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        return Err(parse_err(format!(
            "`{name}`: the last polygon is not closed (its last corner is not a complement)"
        )));
    }
    Ok(polygons)
}

/// One material slot per polygon: `AllSame` unless the layer says `ByPolygon`.
fn material_slots(geometry: &Node, polygon_count: usize) -> Vec<u32> {
    let Some(layer) = geometry.child("LayerElementMaterial") else {
        return vec![0; polygon_count];
    };
    let values = layer.child("Materials").map(Node::array).unwrap_or_default();
    let slot = |at: usize| values.get(at).map(|value| *value as u32).unwrap_or(0);
    if layer.child("MappingInformationType").and_then(Node::text) == Some("ByPolygon") {
        (0..polygon_count).map(slot).collect()
    } else {
        vec![slot(0); polygon_count]
    }
}

/// `Geometry::Cube` → `Cube`; a bare name is kept as it stands.
fn fbx_name(text: &str) -> String {
    match text.rsplit_once("::") {
        Some((_, tail)) if !tail.is_empty() => tail.to_string(),
        _ => text.to_string(),
    }
}

/// Read one `Geometry` node. `None` when it holds no faces at all.
fn read_geometry(geometry: &Node) -> IoResult<Option<MeshData>> {
    let name = geometry
        .props
        .iter()
        .find_map(|prop| match prop {
            Prop::Text(text) => Some(fbx_name(text)),
            Prop::Num(_) => None,
        })
        .unwrap_or_else(|| "mesh".to_string());

    let Some(vertices) = geometry.child("Vertices") else {
        return Ok(None);
    };
    let raw = vertices.array();
    if raw.len() % 3 != 0 {
        return Err(parse_err(format!(
            "`{name}`: Vertices has {} numbers, not a multiple of 3",
            raw.len()
        )));
    }
    let positions: Vec<Vec3> = raw
        .chunks(3)
        .map(|corner| Vec3::new(corner[0] as f32, corner[1] as f32, corner[2] as f32))
        .collect();

    let Some(indices) = geometry.child("PolygonVertexIndex") else {
        return Err(parse_err(format!("`{name}` has Vertices but no PolygonVertexIndex")));
    };
    let polygons = polygons(&indices.array(), positions.len(), &name)?;
    if polygons.is_empty() {
        return Ok(None);
    }

    let normals = read_layer(geometry, "LayerElementNormal", "Normals", "NormalsIndex", 3)?;
    let uvs = read_layer(geometry, "LayerElementUV", "UV", "UVIndex", 2)?;
    let slots = material_slots(geometry, polygons.len());

    // Corner de-duplication, as in the OBJ reader: one output vertex per
    // distinct (position, normal, uv), with the three arrays parallel.
    let mut mesh = Mesh::new();
    let mut map: HashMap<(u32, i64, i64), u32> = HashMap::new();
    let mut corner_at = 0usize;
    for (polygon, corners) in polygons.iter().enumerate() {
        let mut built = Vec::with_capacity(corners.len());
        for &point in corners {
            let point_index = point as usize;
            let normal = match &normals {
                Some(layer) => Some(layer.value_index(corner_at, point_index)?),
                None => None,
            };
            let uv = match &uvs {
                Some(layer) => Some(layer.value_index(corner_at, point_index)?),
                None => None,
            };
            let key = (point, normal.map_or(-1, |at| at as i64), uv.map_or(-1, |at| at as i64));
            let index = *map.entry(key).or_insert_with(|| {
                let index = mesh.positions.len() as u32;
                mesh.positions.push(positions[point_index]);
                if let (Some(layer), Some(at)) = (&normals, normal) {
                    mesh.normals.push(layer.value3(at));
                }
                if let (Some(layer), Some(at)) = (&uvs, uv) {
                    mesh.uvs.push(layer.value2(at));
                }
                index
            });
            built.push(Corner { position: index, normal: index, uv: index });
            corner_at += 1;
        }
        mesh.faces.push(Face { corners: built, material: slots[polygon] });
    }

    Ok(Some(MeshData { name, mesh: finalize_mesh(mesh), material: None }))
}

pub fn import(path: &Path) -> IoResult<LoadedScene> {
    let text = fs::read_to_string(path)?;
    import_str(&text)
}

pub fn import_str(text: &str) -> IoResult<LoadedScene> {
    let roots = parse_tree(text)?;
    let objects = roots
        .iter()
        .find(|node| node.name == "Objects")
        .ok_or_else(|| parse_err("no Objects section"))?;
    // Where the objects are placed: the `Model` objects and the connections
    // between them say so.
    let transforms = object_transforms(
        objects,
        roots.iter().find(|node| node.name == "Connections"),
    );
    let mut meshes = Vec::new();
    for geometry in objects.children.iter().filter(|node| node.name == "Geometry") {
        let mut mesh = match read_geometry(geometry)? {
            Some(mesh) => mesh,
            None => continue,
        };
        // The model's transform is baked in, so the object lands where the
        // file put it.
        if let Some(id) = geometry.props.first().and_then(|prop| match prop {
            Prop::Num(number) => Some(*number as u64),
            _ => None,
        }) {
            if let Some(transform) = transforms.get(&id) {
                mesh.mesh.transform(*transform);
            }
        }
        meshes.push(mesh);
    }
    if meshes.is_empty() {
        return Err(parse_err("no geometry with faces"));
    }
    Ok(LoadedScene { meshes, materials: Vec::new() })
}

/// The transform each object is placed by, from the `Model` objects and the
/// connections between them.
///
/// FBX keeps the geometry in `Geometry` objects and the place in `Model`
/// objects; the `Connections` section links them, naming a child and its
/// parent. The transforms are accumulated down the tree and baked into the
/// vertices, so an imported model lands where the file put it.
fn object_transforms(objects: &Node, connections: Option<&Node>) -> BTreeMap<u64, Mat4> {
    // Each model's own transform.
    let mut local: BTreeMap<u64, Mat4> = BTreeMap::new();
    for model in objects.children.iter().filter(|node| node.name == "Model") {
        let Some(id) = model.props.first().and_then(|prop| match prop {
            Prop::Num(number) => Some(*number as u64),
            _ => None,
        }) else {
            continue;
        };
        local.insert(id, model_transform(model));
    }
    // The parent of each object.
    let mut parent: BTreeMap<u64, u64> = BTreeMap::new();
    if let Some(connections) = connections {
        for connection in connections.children.iter().filter(|node| node.name == "C") {
            let numbers: Vec<u64> = connection
                .props
                .iter()
                .filter_map(|prop| match prop {
                    Prop::Num(number) => Some(*number as u64),
                    _ => None,
                })
                .collect();
            if numbers.len() >= 2 {
                parent.insert(numbers[0], numbers[1]);
            }
        }
    }
    // The transform of every object: a geometry takes the transform of the
    // model that shows it, and a model takes its own and its parents'.
    let mut out: BTreeMap<u64, Mat4> = BTreeMap::new();
    for (id, own) in &local {
        let mut chain = vec![*own];
        let mut current = *id;
        while let Some(next) = parent.get(&current) {
            let Some(node) = local.get(next) else { break };
            chain.push(*node);
            current = *next;
        }
        // A child's transform is in its parent's space, so the parent's comes
        // first.
        let mut m = Mat4::IDENTITY;
        for node in chain.iter().rev() {
            m = m * *node;
        }
        out.insert(*id, m);
    }
    // A geometry is placed by the model that connects to it.
    for (child, parent_id) in &parent {
        if let Some(transform) = out.get(parent_id) {
            out.insert(*child, *transform);
        }
    }
    out
}

/// A model's own transform: its local translation, rotation and scaling.
fn model_transform(model: &Node) -> Mat4 {
    let mut m = Mat4::IDENTITY;
    let Some(properties) = model.child("Properties70") else {
        return m;
    };
    for property in properties.children.iter().filter(|node| node.name == "P") {
        let name = property.text().unwrap_or_default().to_string();
        let values: Vec<f64> = property.numbers();
        match name.as_str() {
            "Lcl Translation" if values.len() >= 3 => {
                m = m * Mat4::from_translation(Vec3::new(
                    values[0] as f32,
                    values[1] as f32,
                    values[2] as f32,
                ));
            }
            "Lcl Rotation" if values.len() >= 3 => {
                // Euler angles in degrees, in the X, Y, Z order the format's
                // default rotation order names: the X turn first.
                let (x, y, z) = (
                    values[0].to_radians() as f32,
                    values[1].to_radians() as f32,
                    values[2].to_radians() as f32,
                );
                m = m * Mat4::from_rotation_z(z)
                    * Mat4::from_rotation_y(y)
                    * Mat4::from_rotation_x(x);
            }
            "Lcl Scaling" if values.len() >= 3 => {
                m = m * Mat4::from_scale(Vec3::new(
                    values[0] as f32,
                    values[1] as f32,
                    values[2] as f32,
                ));
            }
            _ => {}
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Export.
// ---------------------------------------------------------------------------

fn number_list(values: impl Iterator<Item = f32>) -> String {
    values.map(|value| value.to_string()).collect::<Vec<_>>().join(",")
}

fn index_list(values: &[i64]) -> String {
    values.iter().map(|value| value.to_string()).collect::<Vec<_>>().join(",")
}

/// A name safe to put inside the writer's quoted strings.
fn quoted(name: &str) -> String {
    name.replace(['"', '\n', '\r'], "'")
}

pub fn export(path: &Path, data: &MeshData) -> IoResult<()> {
    fs::write(path, export_str(data))?;
    Ok(())
}

pub fn export_str(data: &MeshData) -> String {
    let mesh = &data.mesh;
    let name = quoted(&data.name);
    let mut out = String::from("; FBX 7.4.0 project file\n; Written by OpenForge\n");
    out.push_str("FBXHeaderExtension:  {\n");
    out.push_str("    FBXHeaderVersion: 1003\n");
    out.push_str("    FBXVersion: 7400\n");
    out.push_str(&format!("    Creator: \"OpenForge {}\"\n", env!("CARGO_PKG_VERSION")));
    out.push_str("}\n");
    // Y-up, right-handed, in the file's own units: the geometry is written
    // as the engine holds it, and the app's import options decide scale.
    out.push_str("GlobalSettings:  {\n    Version: 1000\n    Properties70:  {\n");
    out.push_str("        P: \"UpAxis\", \"int\", \"Integer\", \"\",1\n");
    out.push_str("        P: \"UpAxisSign\", \"int\", \"Integer\", \"\",1\n");
    out.push_str("        P: \"FrontAxis\", \"int\", \"Integer\", \"\",2\n");
    out.push_str("        P: \"FrontAxisSign\", \"int\", \"Integer\", \"\",1\n");
    out.push_str("        P: \"CoordAxis\", \"int\", \"Integer\", \"\",0\n");
    out.push_str("        P: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1\n");
    out.push_str("        P: \"UnitScaleFactor\", \"double\", \"Number\", \"\",1\n");
    out.push_str("    }\n}\n");
    out.push_str("Definitions:  {\n    Version: 100\n    Count: 2\n}\n");
    out.push_str("Objects:  {\n");

    out.push_str(&format!("    Geometry: 1, \"Geometry::{name}\", \"Mesh\" {{\n"));
    let vertices: Vec<f32> =
        mesh.positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
    out.push_str(&format!(
        "        Vertices: *{} {{\n            a: {}\n        }}\n",
        vertices.len(),
        number_list(vertices.into_iter())
    ));

    // Each polygon closes with its last corner complemented: `~i` is `-i - 1`.
    let mut indices: Vec<i64> = Vec::new();
    for face in &mesh.faces {
        for (corner, point) in face.corners.iter().enumerate() {
            let index = point.position as i64;
            indices.push(if corner + 1 == face.corners.len() { -index - 1 } else { index });
        }
    }
    out.push_str(&format!(
        "        PolygonVertexIndex: *{} {{\n            a: {}\n        }}\n",
        indices.len(),
        index_list(&indices)
    ));
    out.push_str("        GeometryVersion: 124\n");

    if !mesh.normals.is_empty() {
        let normals: Vec<f32> = mesh
            .faces
            .iter()
            .flat_map(|face| face.corners.iter())
            .flat_map(|corner| {
                let normal = mesh.normals.get(corner.normal as usize).copied().unwrap_or(Vec3::ZERO);
                [normal.x, normal.y, normal.z]
            })
            .collect();
        out.push_str("        LayerElementNormal: 0 {\n");
        out.push_str("            Version: 101\n");
        out.push_str("            Name: \"\"\n");
        out.push_str("            MappingInformationType: \"ByPolygonVertex\"\n");
        out.push_str("            ReferenceInformationType: \"Direct\"\n");
        out.push_str(&format!(
            "            Normals: *{} {{\n                a: {}\n            }}\n",
            normals.len(),
            number_list(normals.into_iter())
        ));
        out.push_str("        }\n");
    }

    if !mesh.uvs.is_empty() {
        let uvs: Vec<f32> = mesh
            .faces
            .iter()
            .flat_map(|face| face.corners.iter())
            .flat_map(|corner| {
                let uv = mesh.uvs.get(corner.uv as usize).copied().unwrap_or(Vec2::ZERO);
                [uv.x, uv.y]
            })
            .collect();
        out.push_str("        LayerElementUV: 0 {\n");
        out.push_str("            Version: 101\n");
        out.push_str("            Name: \"UVMap\"\n");
        out.push_str("            MappingInformationType: \"ByPolygonVertex\"\n");
        out.push_str("            ReferenceInformationType: \"Direct\"\n");
        out.push_str(&format!(
            "            UV: *{} {{\n                a: {}\n            }}\n",
            uvs.len(),
            number_list(uvs.into_iter())
        ));
        out.push_str("        }\n");
    }

    // Material slots: one value for all faces, unless they differ.
    let slots: Vec<u32> = mesh.faces.iter().map(|face| face.material).collect();
    let all_same = slots.iter().all(|slot| *slot == slots.first().copied().unwrap_or(0));
    out.push_str("        LayerElementMaterial: 0 {\n");
    out.push_str("            Version: 101\n");
    out.push_str("            Name: \"\"\n");
    out.push_str(&format!(
        "            MappingInformationType: \"{}\"\n",
        if all_same { "AllSame" } else { "ByPolygon" }
    ));
    out.push_str("            ReferenceInformationType: \"IndexToDirect\"\n");
    let materials: Vec<i64> = if all_same {
        vec![slots.first().copied().unwrap_or(0) as i64]
    } else {
        slots.iter().map(|slot| *slot as i64).collect()
    };
    out.push_str(&format!(
        "            Materials: *{} {{\n                a: {}\n            }}\n",
        materials.len(),
        index_list(&materials)
    ));
    out.push_str("        }\n");
    out.push_str("    }\n");

    out.push_str(&format!(
        "    Model: 2, \"Model::{name}\", \"Mesh\" {{\n        Version: 232\n        Properties70:  {{\n            P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,0,0\n            P: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\",0,0,0\n            P: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A\",1,1,1\n        }}\n    }}\n"
    ));
    out.push_str("}\n");
    out.push_str("Connections:  {\n    C: \"OO\",1,2\n    C: \"OO\",2,0\n}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal a document earns, as a string.
    fn refusal(text: &str) -> String {
        match import_str(text) {
            Ok(loaded) => panic!("expected a refusal, got {} mesh(es)", loaded.mesh_count()),
            Err(error) => error.to_string(),
        }
    }

    /// A small realistic document: a quad and a triangle sharing positions, a
    /// direct per-polygon-vertex normal layer, a UV layer through its index
    /// array, per-polygon material slots, and a `Model` carrying a transform.
    const DOCUMENT: &str = r#"; FBX 7.4.0 project file
FBXHeaderExtension:  {
    FBXHeaderVersion: 1003
    FBXVersion: 7400
}
Objects:  {
    Geometry: 100, "Geometry::Panel", "Mesh" {
        Vertices: *18 {
            a: 0,0,0,1,0,0,1,1,0,
               0,0,0,
               0,1,0,2,0,0
        }
        PolygonVertexIndex: *7 {
            a: 0,1,2,-4,0,4,~5
        }
        LayerElementNormal: 0 {
            Version: 101
            Name: ""
            MappingInformationType: "ByPolygonVertex"
            ReferenceInformationType: "Direct"
            Normals: *21 {
                a: 0,0,1,0,0,1,0,0,1,
                   0,0,1,0,0,1,0,0,1,
                   0,0,1
            }
        }
        LayerElementUV: 0 {
            Version: 101
            Name: "UVMap"
            MappingInformationType: "ByPolygonVertex"
            ReferenceInformationType: "IndexToDirect"
            UV: *8 {
                a: 0,0,1,0,1,1,0,1
            }
            UVIndex: *7 {
                a: 0,1,2,3,0,2,3
            }
        }
        LayerElementMaterial: 0 {
            Version: 101
            Name: ""
            MappingInformationType: "ByPolygon"
            ReferenceInformationType: "IndexToDirect"
            Materials: *2 {
                a: 0,1
            }
        }
        GeometryVersion: 124
    }
    Model: 200, "Model::Panel", "Mesh" {
        Version: 232
        Properties70:  {
            P: "Lcl Translation", "Lcl Translation", "", "A",1,2,3
        }
    }
}
Connections:  {
    C: "OO",100,200
    C: "OO",200,0
}
"#;

    #[test]
    fn a_real_document_reads_its_polygons_layers_and_slots() {
        let loaded = import_str(DOCUMENT).unwrap();
        assert_eq!(loaded.mesh_count(), 1);
        assert_eq!(loaded.meshes[0].name, "Panel");
        let mesh = &loaded.meshes[0].mesh;

        // Two polygons, split where the complements are: a quad and a
        // triangle, seven corners with the shared corners kept apart.
        assert_eq!(mesh.face_count(), 2);
        assert_eq!(mesh.faces[0].corners.len(), 4);
        assert_eq!(mesh.faces[1].corners.len(), 3);
        assert_eq!(mesh.vertex_count(), 7, "one vertex per distinct corner triple");

        // The model's transform is baked in: the document's `Lcl Translation`
        // of (1, 2, 3) moved the geometry with it.
        assert_eq!(mesh.positions[1], Vec3::new(2.0, 2.0, 3.0));
        assert_eq!(
            mesh.positions[5],
            Vec3::new(1.0, 3.0, 3.0),
            "the triangle's middle point"
        );
        // The Direct normal layer is per corner; every one is +Z here.
        assert_eq!(mesh.normals.len(), 7);
        assert!(mesh.normals.iter().all(|normal| *normal == Vec3::Z));
        // The UV layer goes through UVIndex: the quad's corners get 0..3 and
        // the triangle repeats 0, 2 and 3.
        assert_eq!(mesh.uvs[0], Vec2::new(0.0, 0.0));
        assert_eq!(mesh.uvs[3], Vec2::new(0.0, 1.0));
        assert_eq!(mesh.uvs[5], Vec2::new(1.0, 1.0), "the triangle's middle corner");
        // Per-polygon slots, and the shared control point did *not* merge the
        // two polygons' corners.
        assert_eq!([mesh.faces[0].material, mesh.faces[1].material], [0, 1]);
        assert_eq!(mesh.faces[0].corners[0].position, 0);
        assert_eq!(mesh.faces[1].corners[0].position, 4);
    }

    #[test]
    fn wrapped_lists_and_comments_are_read() {
        let text = "; a wrapped document\nObjects:  {\n    Geometry: 7, \"Geometry::Tri\", \"Mesh\" {\n        Vertices: *9 {\n            a: 0,0,0,\n; right in the middle of the values\n               1,0,0,\n               0,1,0\n        }\n        PolygonVertexIndex: *3 {\n            a: 0,1,-3\n        }\n    }\n}\n";
        let loaded = import_str(text).unwrap();
        let mesh = &loaded.meshes[0].mesh;
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.positions[2], Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(mesh.face_count(), 1);
    }

    #[test]
    fn control_point_layers_map_by_vertex() {
        let text = "Objects:  {\n    Geometry: 1, \"Geometry::Tri\", \"Mesh\" {\n        Vertices: *9 {\n            a: 0,0,0,1,0,0,0,1,0\n        }\n        PolygonVertexIndex: *3 {\n            a: 0,1,-3\n        }\n        LayerElementUV: 0 {\n            MappingInformationType: \"ByControlPoint\"\n            ReferenceInformationType: \"Direct\"\n            UV: *6 {\n                a: 0,0,1,0,0,1\n            }\n        }\n    }\n}\n";
        let loaded = import_str(text).unwrap();
        let mesh = &loaded.meshes[0].mesh;
        assert_eq!(mesh.uvs, vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)]);
        assert!(mesh.normals.len() == 3, "no normal layer: smooth ones were computed");
    }

    #[test]
    fn several_geometries_become_several_meshes() {
        let text = "Objects:  {\n    Geometry: 1, \"Geometry::Left\", \"Mesh\" {\n        Vertices: *9 { a: 0,0,0,1,0,0,0,1,0 }\n        PolygonVertexIndex: *3 { a: 0,1,-3 }\n    }\n    Geometry: 2, \"Geometry::Right\", \"Mesh\" {\n        Vertices: *9 { a: 2,0,0,3,0,0,2,1,0 }\n        PolygonVertexIndex: *3 { a: 0,1,-3 }\n    }\n}\n";
        let loaded = import_str(text).unwrap();
        assert_eq!(loaded.mesh_count(), 2);
        assert_eq!(loaded.meshes[0].name, "Left");
        assert_eq!(loaded.meshes[1].name, "Right");
        assert_eq!(loaded.meshes[1].mesh.positions[0], Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn a_quad_round_trips_through_the_writer() {
        let mut mesh = Mesh::new();
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            mesh.add_vertex(Vec3::new(x, y, 0.0));
        }
        mesh.add_face(&[0, 1, 2, 3]);
        // `add_face` seeded one default slot each; replace them with one
        // normal and one UV per position, with the corners pointing at their
        // own slot: the engine's parallel-arrays convention.
        mesh.normals.clear();
        mesh.uvs.clear();
        for _ in 0..4 {
            mesh.normals.push(Vec3::Z);
        }
        for (u, v) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            mesh.uvs.push(Vec2::new(u, v));
        }
        for corner in mesh.faces[0].corners.iter_mut() {
            corner.normal = corner.position;
            corner.uv = corner.position;
        }
        let data = MeshData { name: "quad".into(), mesh, material: None };

        let text = export_str(&data);
        assert!(text.starts_with("; FBX"), "{text}");
        assert!(text.contains("FBXVersion: 7400"), "{text}");
        assert!(text.contains("\"Geometry::quad\""), "{text}");
        assert!(text.contains("a: 0,1,2,-4"), "the quad closes with a complement: {text}");
        assert!(text.contains("MappingInformationType: \"ByPolygonVertex\""));
        assert!(text.contains("P: \"UpAxis\""), "the file names its axes");

        let loaded = import_str(&text).unwrap();
        assert_eq!(loaded.mesh_count(), 1);
        assert_eq!(loaded.meshes[0].name, "quad");
        let mesh = &loaded.meshes[0].mesh;
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.faces[0].corners.len(), 4);
        assert_eq!(mesh.positions[2], Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(mesh.uvs[3], Vec2::new(0.0, 1.0));
        assert_eq!(mesh.normals[0], Vec3::Z);
    }

    #[test]
    fn ngons_and_material_slots_survive_a_round_trip() {
        let mut mesh = Mesh::new();
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (1.0, 2.0), (0.0, 1.0)] {
            mesh.add_vertex(Vec3::new(x, y, 0.5));
        }
        mesh.add_face(&[0, 1, 2, 3, 4]);
        for (x, y) in [(3.0, 0.0), (4.0, 0.0), (4.0, 1.0), (3.0, 1.0)] {
            mesh.add_vertex(Vec3::new(x, y, 0.5));
        }
        mesh.add_face(&[5, 6, 7, 8]);
        mesh.faces[1].material = 2;
        let data = MeshData { name: "patch".into(), mesh, material: None };

        let text = export_str(&data);
        assert!(text.contains("MappingInformationType: \"ByPolygon\""), "{text}");
        let loaded = import_str(&text).unwrap();
        let mesh = &loaded.meshes[0].mesh;
        assert_eq!(mesh.face_count(), 2);
        assert_eq!(mesh.faces[0].corners.len(), 5, "the pentagon stayed one face");
        assert_eq!(mesh.faces[1].corners.len(), 4);
        assert_eq!([mesh.faces[0].material, mesh.faces[1].material], [0, 2], "slots ride the file");
    }

    #[test]
    fn a_single_material_layer_is_written_all_same() {
        let mut mesh = Mesh::new();
        mesh.add_vertex(Vec3::ZERO);
        mesh.add_vertex(Vec3::X);
        mesh.add_vertex(Vec3::Y);
        mesh.add_face(&[0, 1, 2]);
        let data = MeshData { name: "tri".into(), mesh, material: None };
        let text = export_str(&data);
        assert!(text.contains("MappingInformationType: \"AllSame\""), "{text}");
        let loaded = import_str(&text).unwrap();
        assert_eq!(loaded.meshes[0].mesh.faces[0].material, 0);
    }

    #[test]
    fn broken_documents_are_refused() {
        let cases = [
            ("", "no Objects section"),
            ("; only a comment\n", "no Objects section"),
            ("Objects:  {\n}\n", "no geometry with faces"),
            (
                "Objects:  {\n    Geometry: 1, \"Geometry::Bad\", \"Mesh\" {\n        Vertices: *2 {\n            a: 0,0\n        }\n        PolygonVertexIndex: *3 {\n            a: 0,1,-1\n        }\n    }\n}\n",
                "not a multiple of 3",
            ),
            (
                "Objects:  {\n    Geometry: 1, \"Geometry::Bad\", \"Mesh\" {\n        Vertices: *9 {\n            a: 0,0,0,1,0,0,0,1,0\n        }\n        PolygonVertexIndex: *3 {\n            a: 0,1,-9\n        }\n    }\n}\n",
                "names vertex 8 of 3",
            ),
            (
                "Objects:  {\n    Geometry: 1, \"Geometry::Bad\", \"Mesh\" {\n        Vertices: *9 {\n            a: 0,0,0,1,0,0,0,1,0\n        }\n        PolygonVertexIndex: *2 {\n            a: 0,-2\n        }\n    }\n}\n",
                "need at least 3",
            ),
            (
                "Objects:  {\n    Geometry: 1, \"Geometry::Bad\", \"Mesh\" {\n        Vertices: *9 {\n            a: 0,0,0,1,0,0,0,1,0\n        }\n        PolygonVertexIndex: *2 {\n            a: 0,1\n        }\n    }\n}\n",
                "not closed",
            ),
            (
                "Objects:  {\n    Geometry: 1, \"Geometry::Bad\", \"Mesh\" {\n",
                "unterminated",
            ),
        ];
        for (text, message) in cases {
            let error = refusal(text);
            assert!(error.contains(message), "{error:?} should mention {message:?}");
        }
    }

    #[test]
    fn a_layer_that_does_not_cover_its_corners_is_refused() {
        // Two corners, one UV: IndexToDirect runs off the end of the array.
        let text = "Objects:  {\n    Geometry: 1, \"Geometry::Bad\", \"Mesh\" {\n        Vertices: *9 {\n            a: 0,0,0,1,0,0,0,1,0\n        }\n        PolygonVertexIndex: *3 {\n            a: 0,1,-3\n        }\n        LayerElementUV: 0 {\n            MappingInformationType: \"ByPolygonVertex\"\n            ReferenceInformationType: \"IndexToDirect\"\n            UV: *2 {\n                a: 0,0\n            }\n            UVIndex: *1 {\n                a: 0\n            }\n        }\n    }\n}\n";
        let error = refusal(text);
        assert!(error.contains("no entry 1"), "{error:?}");
    }
    /// A document with one triangle and the models that place it.
    fn placed_document(models: &str, connections: &str) -> String {
        format!(
            "; FBX 7.4.0 project file
Objects:  {{
    Geometry: 100, \"Geometry::Tri\", \"Mesh\" {{
        Vertices: *9 {{ a: 0,0,0, 1,0,0, 0,1,0 }}
        PolygonVertexIndex: *3 {{ a: 0,1,-3 }}
    }}
{models}}}
Connections:  {{
{connections}
}}
"
        )
    }

    #[test]
    fn a_models_transform_is_baked_into_the_imported_mesh() {
        let text = placed_document(
            "    Model: 200, \"Model::Tri\", \"Mesh\" {\n        Version: 232\n        Properties70:  {\n            P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",5,0,0\n        }\n    }\n",
            "    C: \"OO\",100,200\n",
        );
        let loaded = import_str(&text).unwrap();
        let mesh = &loaded.meshes[0].mesh;
        assert_eq!(mesh.vertex_count(), 3);
        assert!(
            (mesh.positions[1] - Vec3::new(6.0, 0.0, 0.0)).length() < 1e-5,
            "the vertex moved with the model: {:?}",
            mesh.positions[1]
        );
    }

    #[test]
    fn a_models_rotation_is_baked_into_the_imported_mesh() {
        // A quarter turn about Z: the triangle's far corner goes up.
        let text = placed_document(
            "    Model: 200, \"Model::Tri\", \"Mesh\" {\n        Version: 232\n        Properties70:  {\n            P: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\",0,0,90\n        }\n    }\n",
            "    C: \"OO\",100,200\n",
        );
        let loaded = import_str(&text).unwrap();
        let mesh = &loaded.meshes[0].mesh;
        assert!(
            (mesh.positions[1] - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-5,
            "the vertex turned with the model: {:?}",
            mesh.positions[1]
        );
    }

    #[test]
    fn nested_models_transforms_are_accumulated() {
        // A model inside another model: the child's transform rides on the
        // parent's.
        let text = placed_document(
            "    Model: 200, \"Model::Tri\", \"Mesh\" {\n        Version: 232\n        Properties70:  {\n            P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",5,0,0\n        }\n    }\n    Model: 300, \"Model::Group\", \"Model\" {\n        Version: 232\n        Properties70:  {\n            P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,0,10\n        }\n    }\n",
            "    C: \"OO\",100,200\n    C: \"OO\",200,300\n",
        );
        let loaded = import_str(&text).unwrap();
        let mesh = &loaded.meshes[0].mesh;
        assert!(
            (mesh.positions[1] - Vec3::new(6.0, 0.0, 10.0)).length() < 1e-5,
            "the vertex sits under both models: {:?}",
            mesh.positions[1]
        );
    }

    #[test]
    fn a_geometry_with_no_model_stays_where_it_was() {
        let text = placed_document("", "");
        let loaded = import_str(&text).unwrap();
        let mesh = &loaded.meshes[0].mesh;
        assert!((mesh.positions[1] - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
    }

}
