export const MetadataSection = ({
  title,
  data,
}: {
  title: string;
  data: any;
}) => {
  if (!data || Object.keys(data).length === 0) return null;

  return (
    <div className="mb-2">
      <h4 className="font-semibold text-xs text-gray-700 mb-1">{title}</h4>
      <div className="border border-gray-200 p-2 text-xs bg-white">
        <pre className="whitespace-pre-wrap overflow-x-auto">
          {JSON.stringify(data, null, 1)}
        </pre>
      </div>
    </div>
  );
};
