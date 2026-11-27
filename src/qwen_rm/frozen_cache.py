# Resolve exact JPEG inputs from an audited, retained frozen cache.
import hashlib,json,os
from pathlib import Path

class VerifiedFrames:
    def __init__(self):
        index=Path(os.environ['FRAME_RM_VERIFIED_CACHE_INDEX']).resolve(strict=True)
        payload=index.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=os.environ['FRAME_RM_VERIFIED_CACHE_INDEX_SHA256']:
            raise ValueError('verified_media_index_changed')
        data=json.loads(payload)
        if data['schema']!='verified-frozen-RGB-frame-index-v1':
            raise ValueError('unsupported_verified_media_index')
        if data['snapshot_sha256']!=os.environ['FRAME_RM_VERIFIED_SNAPSHOT_SHA256']:
            raise ValueError('verified_cache_snapshot_mismatch')
        self.root=index.parent/'media'
        self.roots={Path(p).resolve(strict=True) for p in data.get('retained_roots',[str(self.root)])}
        self.media=data['media']
        self.checked={}

    def paths(self,row):
        key=row['media_key']
        if len(key)!=64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('invalid_media_cache_key')
        item=self.media.get(key)
        if item is None:
            raise ValueError('media_not_in_verified_index:'+key)
        for field in ('video','video_bytes','video_mtime_ns'):
            if row[field]!=item[field]:
                raise ValueError('verified_media_metadata_mismatch:'+field)
        if row['case_id'] not in item['cases']:
            raise ValueError('case_not_in_verified_index')
        paths=[]
        for frame in row['frames']:
            entry=item['frames'].get(str(frame))
            if entry is None:
                raise ValueError('requested_frame_not_in_verified_index:'+str(frame))
            path=Path(entry['path'])
            retained=Path(item.get('retained_media_root',str(self.root)))
            if retained.resolve(strict=True) not in self.roots:
                raise ValueError('unapproved_retained_cache_root')
            expected=retained/key/('frame_%06d.jpg'%frame)
            if path!=expected or path.is_symlink():
                raise ValueError('verified_frame_path_mismatch')
            stat=path.stat()
            signature=(stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if stat.st_size!=entry['bytes']:
                raise ValueError('verified_frame_size_changed')
            if self.checked.get(str(path))!=signature:
                if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                    raise ValueError('verified_frame_bytes_changed')
                self.checked[str(path)]=signature
            paths.append(path)
        return paths

_reader=None
def verified_paths(row):
    global _reader
    if _reader is None:
        _reader=VerifiedFrames()
    return _reader.paths(row)
